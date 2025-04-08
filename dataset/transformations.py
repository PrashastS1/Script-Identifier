import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch

class LanguageRecognitionTransforms:
    
    @staticmethod
    def get_transforms(backbone_type: str = None, phase: str = 'train', img_size: int = 224):
        """
        Args:
            backbone_type: None (custom), 'resnet50', 'vgg', 'vit', 'swin', 'beit', 'sift', 'hog'
            phase: 'train' or 'test'
            img_size: Final output size (square)
        """

        # Normalization values for different backbones
        norms = {
            'imagenet': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
            'vit': {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]},
            'default': {'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}
        }
        
        # Base parameters for transformations
        base_params = {
            'border_mode': cv2.BORDER_CONSTANT,
            'mask_value': 255,
            'interpolation': cv2.INTER_CUBIC
        }

        # Test phase transformations
        if phase == 'test':
            if backbone_type in ['sift', 'hog']:
                return A.Compose([
                    A.LongestMaxSize(img_size + 32, interpolation=base_params['interpolation']),
                    A.PadIfNeeded(min_height=img_size, min_width=img_size, 
                                border_mode=base_params['border_mode']),
                    A.CenterCrop(img_size, img_size),
                    A.Lambda(image=lambda x, **kwargs: torch.from_numpy(x).permute(2,0,1).float())
                ])
            else:
                norm = norms['imagenet' if backbone_type in ['resnet50', 'vgg', 'vit'] else 'default']
                return A.Compose([
                    A.LongestMaxSize(img_size + 32, interpolation=base_params['interpolation']),
                    A.PadIfNeeded(min_height=img_size, min_width=img_size, 
                                border_mode=base_params['border_mode']),
                    A.CenterCrop(img_size, img_size),
                    A.Normalize(**norm),
                    ToTensorV2()
                ])

        transforms = []
        crop_size = int(img_size * 1.2)
        
        # Common initial transforms for all train phases
        transforms.extend([
            A.LongestMaxSize(crop_size, interpolation=base_params['interpolation']),
            A.PadIfNeeded(crop_size, crop_size)
        ])

        # Backbone-specific transforms
        if not backbone_type or backbone_type == 'custom':
            transforms.extend([
                A.RandomResizedCrop((img_size, img_size), scale=(0.85, 1)),
                A.HorizontalFlip(p=0.4),
                A.Affine(
                    translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
                    scale=(0.9, 1.1),
                    rotate=(-30, 30),
                    shear=0,
                    interpolation=cv2.INTER_CUBIC,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=255,
                    fit_output=False,
                    keep_ratio=False,
                    p=0.5
                ),
                A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05, p=0.5),
                A.CoarseDropout(
                    num_holes_range=(3, 6),
                    hole_height_range=(0.05, 0.1),
                    hole_width_range=(0.05, 0.1),
                    fill=255,
                    p=0.5
                ),
                A.Normalize(**norms['default']),
                ToTensorV2()
            ])

        elif backbone_type in ['resnet50', 'vgg']:
            transforms.extend([
                A.RandomResizedCrop((img_size, img_size), scale=(0.75, 1)),
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
                    scale=(0.9, 1.1),
                    rotate=(-30, 30),
                    shear=0,
                    interpolation=cv2.INTER_CUBIC,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=255,
                    fit_output=False,
                    keep_ratio=False,
                    p=0.5
                ),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.4),
                A.CoarseDropout(
                    num_holes_range=(3, 6),
                    hole_height_range=(0.05, 0.1),
                    hole_width_range=(0.05, 0.1),
                    fill=255,
                    p=0.5
                ),
                A.Normalize(**norms['imagenet']),
                ToTensorV2()
            ])

        elif backbone_type in ['vit']:
            transforms.extend([
                A.RandomResizedCrop((img_size, img_size), scale=(0.7, 1), ratio=(0.8, 1.2)),
                A.HorizontalFlip(p=0.3),
                A.Affine(
                    translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
                    scale=(0.9, 1.1),
                    rotate=(-30, 30),
                    shear=0,
                    interpolation=cv2.INTER_CUBIC,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=255,
                    fit_output=False,
                    keep_ratio=False,
                    p=0.5
                ),
                A.RandomGamma(gamma_limit=(90, 110), p=0.4),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.CoarseDropout(
                    num_holes_range=(3, 6),
                    hole_height_range=(0.05, 0.1),
                    hole_width_range=(0.05, 0.1),
                    fill=255,
                    p=0.5
                ),
                A.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.2, hue=0.05, p=0.7),
                A.Normalize(**norms['imagenet']),
                ToTensorV2()
            ])

        elif backbone_type in ['sift', 'hog']:
            transforms.extend([
                A.RandomResizedCrop((img_size, img_size), scale=(0.85, 1)),
                A.HorizontalFlip(p=0.4),
                A.Affine(
                    translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
                    scale=(0.9, 1.1),
                    rotate=(-30, 30),
                    shear=0,
                    interpolation=cv2.INTER_CUBIC,
                    border_mode=cv2.BORDER_CONSTANT,
                    fill=255,
                    fit_output=False,
                    keep_ratio=False,
                    p=0.5
                ),
                A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05, p=0.5),
                A.CoarseDropout(
                    num_holes_range=(3, 6),
                    hole_height_range=(0.05, 0.1),
                    hole_width_range=(0.05, 0.1),
                    fill=255,
                    p=0.5
                ),
                A.Lambda(image=lambda x, **kwargs: torch.from_numpy(x).permute(2,0,1).float())
            ])

        return A.Compose(transforms, is_check_shapes=False)

    @staticmethod
    def get_advanced_transforms(backbone_type: str, phase: str, img_size: int = 224):
        raise NotImplementedError("Advanced transforms not implemented yet.")
        # """Includes MixUp/CutMix for transformer models (requires label)"""
        # base_transform = LanguageRecognitionTransforms.get_transforms(backbone_type, phase, img_size)
        
        # if phase == 'train' and backbone_type in ['vit']:
        #     return A.Compose([
        #         base_transform,
        #         A.OneOf([
        #             A.CutMix(num_classes=14, p=0.3),
        #             A.MixUp(num_classes=14, p=0.3)
        #         ], p=0.4)
        #     ])
        # return base_transform
