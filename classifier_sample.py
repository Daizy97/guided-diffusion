"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import (
    resnet50, ResNet50_Weights,
    inception_v3, Inception_V3_Weights,
    vit_b_16, ViT_B_16_Weights
)
from datetime import datetime

# set random seeds
seed = 123
th.manual_seed(seed)
np.random.seed(seed)

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
    attack_target_model_defaults
)


def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure(dir='./logs')  # 之后文件名要加上target model和attack名

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    logger.log(f"loading target model: {args.target_model}\n")
    if args.target_model == 'resnet50':
        target_classifier = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(dist_util.dev())
        input_size = 224
    elif args.target_model == 'inceptionv3':
        target_classifier = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(dist_util.dev())
        input_size = 299
    elif args.target_model == 'vit':
        target_classifier = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(dist_util.dev())
        input_size = 224
    else:
        raise ValueError(f'Unsupported target model name: {args.target_model}')

    target_classifier.eval()
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),  # 调整图片大小，因为预训练模型通常需要224x224的输入
        transforms.ToTensor(),  # 将图片转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])
    to_pil = transforms.ToPILImage()

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log(f"sampling {args.num_samples} images...")
    all_images = []
    all_labels = []
    corrects = 0
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        classes = th.randint(
            low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
        )
        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample_ori = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )
        sample_ori = ((sample_ori + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample_ori.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

        # test the accuracy of the generated samples
        num = 0
        for i in range(sample_ori.shape[0]):
            image = to_pil(sample_ori[i])
            image = transform(image)
            image = image.unsqueeze(0).to(dist_util.dev())
            output = target_classifier(image)
            pred = output.argmax(dim=1, keepdim=True)
            num += th.equal(pred.squeeze(), classes[i])
        corrects += num
        logger.log('Batch accuracy: {}/{}'.format(num, sample_ori.shape[0]))


    logger.log('Accuracy: {}/{} ({:.1f}%)'.format(corrects, args.num_samples, 100. * corrects / args.num_samples))

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=4,
        batch_size=2,
        use_ddim=False,
        model_path="models/256x256_diffusion.pt",
        classifier_path="models/256x256_classifier.pt",
        classifier_scale=1.0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    defaults.update(attack_target_model_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    start = datetime.now()

    main()

    end = datetime.now()
    time = (end-start).seconds
    hours, remainder = divmod(time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.log('Total time: {} hours {} minutes {} seconds'.format(hours, minutes, seconds))
