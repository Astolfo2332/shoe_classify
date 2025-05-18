import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from src.utils.engine import set_seed

def get_model(model_name: str) -> nn.Module:

    if model_name == "tinyvgg":
        model = TinyVGG(input_shape=1, hidden_units=64, output_shape=2)
        return model

    if model_name == "cnnconnecteddeep":
        model = CNNConnectedDeep()
        return model
    
    if "custom" in model_name:
        model_name = model_name.split("_")[1]
        if model_name == "effntb0":
            model = create_effntb0()
            model = CustomEfficientNet(model)
            return model
        if model_name == "effntb2":
            model = create_effntb2()
            # model = CustomEfficientNet(model)
            return model
        if model_name == "convnextTiny":
            model = create_convnext_tiny()
            model = CustomEfficientNet(model)
            return model
        if model_name == "convnextSmall":
            model = create_convnext_small()
            model = CustomEfficientNet(model)
            return model
        if model_name == "resnet50":
            model = create_resnet50()
            model = CustomEfficientNet(model)
            return model
        if model_name == "resnet101":
            model = create_resnet101()
            # model = CustomEfficientNet(model)
            return model
        if model_name == "resnet152":
            model = create_resnet152()
            # model = CustomEfficientNet(model)
            return model

    match model_name:
        case "vit_b_16":
            return create_vit_b_16()

    raise NameError("Ingrese un nombre valido")

class CNNConnectedDeep(nn.Module):
    def __init__(self, num_classes=3):
        super(CNNConnectedDeep, self).__init__()

        # Primeras capas
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)

        # Concatenación de conv1 y conv3
        # 32 (resized conv1) + 64 = 96
        # Bloques de compresión adicionales con conexiones
        self.conv4 = nn.Conv2d(96, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 160, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(160)

        self.conv6 = nn.Conv2d(160, 192, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(192)

        self.conv7 = nn.Conv2d(192 + 128, 224, kernel_size=3, padding=1)  # concat con out4
        self.bn7 = nn.BatchNorm2d(224)

        self.conv8 = nn.Conv2d(224 + 160, 256, kernel_size=3, padding=1)  # concat con out5
        self.bn8 = nn.BatchNorm2d(256)

        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))  # reduce a [B, 256, 4, 4]

        # Flatten: 256 * 4 * 4 = 4096 → muy alto → reducimos
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)))  # [B, 32, 224, 224]
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out2 = out1 + out2  # Residual connection
        out2 = self.pool(out2)  # [B, 32, 112, 112]

        out3 = F.relu(self.bn3(self.conv3(out2)))
        out3 = self.pool(out3)  # [B, 64, 56, 56]

        out1_resized = F.interpolate(out1, size=out3.shape[2:])
        concat1 = torch.cat((out3, out1_resized), dim=1)  # [B, 96, 56, 56]

        # Bloque 4
        out4 = F.relu(self.bn4(self.conv4(concat1)))
        out4 = self.pool(out4)  # [B, 128, 28, 28]

        # Bloque 5
        out5 = F.relu(self.bn5(self.conv5(out4)))
        out5 = self.pool(out5)  # [B, 160, 14, 14]

        # Bloque 6
        out6 = F.relu(self.bn6(self.conv6(out5)))
        out6 = self.pool(out6)  # [B, 192, 7, 7]

        # Concat out4 (resized) con out6
        out4_resized = F.interpolate(out4, size=out6.shape[2:])
        concat2 = torch.cat((out6, out4_resized), dim=1)  # [B, 192+128=320, 7, 7]
        out7 = F.relu(self.bn7(self.conv7(concat2)))

        # Concat out5 (resized) con out7
        out5_resized = F.interpolate(out5, size=out7.shape[2:])
        concat3 = torch.cat((out7, out5_resized), dim=1)  # [B, 224+160=384, 7, 7]
        out8 = F.relu(self.bn8(self.conv8(concat3)))

        x = self.global_pool(out8)  # [B, 256, 4, 4]
        x = x.view(x.size(0), -1)   # Flatten → [B, 4096]

        x = self.dropout(F.relu(self.fc1(x)))  # 4096 → 512
        x = self.dropout(F.relu(self.fc2(x)))  # 512 → 128
        out = self.fc3(x)  # 128 → num_classes

        return out



class TinyVGG(nn.Module):
    """
    Model architecture copying TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv3d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1), # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv3d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv3d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.LayerNorm((7683200,), elementwise_affine=False, eps=1e-5),
            nn.Linear(in_features= 7683200,
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x


class PreprocessInput(nn.Module):
    def __init__(self):
        super(PreprocessInput, self).__init__()

        self.conv_block_1 = nn.Sequential(
        nn.Conv3d(1, 100, kernel_size=2),  # Remapear de 1 canal a 3 con la combinación de los otros
        nn.ReLU(),
        nn.Conv3d(100, 50, kernel_size=2),  # Remapear de 1 canal a 3 con la combinación de los otros
        nn.ReLU(),
        nn.MaxPool3d(kernel_size=2, stride=1)
        )

        self.conv_block_2 = nn.Sequential(
        nn.Conv3d(50, 25, kernel_size=2),  # Remapear de 1 canal a 3 con la combinación de los otros
        nn.ReLU(),
        nn.Conv3d(25, 13, kernel_size=2),  # Remapear de 1 canal a 3 con la combinación de los otros
        nn.ReLU(),
        nn.MaxPool3d(kernel_size=2, stride=1)
        )

        self.conv_block_3 = nn.Sequential(
        nn.Conv3d(13, 3, kernel_size=2),  # Remapear de 1 canal a 3 con la combinación de los otros
        nn.ReLU()
        )

        self.flatten = nn.Flatten(start_dim=3)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((244, 244))


    def forward(self, x):
        x = self.conv_block_1(x)  
        x = self.conv_block_2(x)  
        x = self.conv_block_3(x)  
        x = self.flatten(x)  
        x = self.adaptive_pool(x)  
        return x

class CustomEfficientNet(nn.Module):
    def __init__(self, base_model):
        super(CustomEfficientNet, self).__init__()
        self.preprocess = PreprocessInput()  # Capa de preprocesamiento
        self.model = base_model  # Modelo EfficientNet

    def forward(self, x):
        x = self.preprocess(x)  # Preprocesar entrada
        x = self.model(x)  # Pasar por EfficientNet
        return x

def create_effntb0() -> nn.Module: 
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights)
    model = freeze_parameters(model)
    print("[INFO] create new effntb0 model.")
    return model

def freeze_parameters(model: nn.Module) -> nn.Module:
    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True), #We dont change this variable
        torch.nn.Linear(in_features=1280, out_features=3, bias=True)
    )
    return model

def create_effntb2():
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    model = torchvision.models.efficientnet_b2(weights=weights)
    for param in model.features.parameters():
        param.requires_grad = False

    set_seed()

    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.3, inplace=True), #We dont change this variable
        torch.nn.Linear(in_features=1408, out_features=3, bias=True)
    )
    print("[INFO] create new effntb2 model.")
    return model

def create_convnext_tiny():
    weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
    model = torchvision.models.convnext_tiny(weights=weights)
    for param in model.features.parameters():
        param.requires_grad = False

    set_seed()

    model.classifier = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.LayerNorm((768,), eps=1e-06, elementwise_affine=True),
        torch.nn.Flatten(start_dim=1, end_dim=-1),
        torch.nn.Linear(in_features=768, out_features=2, bias=True)
    )
    print("[INFO] create new convnext_tiny model.")
    return model

def create_vit_b_16():
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    model = torchvision.models.vit_b_16(weights=weights)
    for param in model.parameters():
        param.requires_grad = False

    set_seed()

    model.heads = nn.Sequential(
        nn.Linear(in_features=768, out_features=256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 3)
    )
    print("[INFO] create new vit_b_16 model.")
    return model

def create_convnext_small():
    weights = torchvision.models.ConvNeXt_Small_Weights.DEFAULT
    model = torchvision.models.convnext_small(weights=weights)
    for param in model.features.parameters():
        param.requires_grad = False

    set_seed()

    model.classifier = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.LayerNorm((768,), eps=1e-06, elementwise_affine=True),
        torch.nn.Flatten(start_dim=1, end_dim=-1),
        torch.nn.Linear(in_features=768, out_features=2, bias=True)
    )
    print("[INFO] create new convnext_small model.")
    return model

def create_resnet50():
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    model = torchvision.models.resnet50(weights=weights)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True), #We dont change this variable
        torch.nn.Linear(in_features=model.fc.in_features, out_features=3, bias=True)
    )
    print("[INFO] create new resnet50 model.")
    return model

def create_resnet101():
    weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V2
    model = torchvision.models.resnet101(weights=weights)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True), #We dont change this variable
        torch.nn.Linear(in_features=model.fc.in_features, out_features=3, bias=True)
    )
    print("[INFO] create new resnet101 model.")
    return model

def create_resnet152():
    weights = torchvision.models.ResNet152_Weights.IMAGENET1K_V2
    model = torchvision.models.resnet152(weights=weights)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True), #We dont change this variable
        torch.nn.Linear(in_features=model.fc.in_features, out_features=3, bias=True)
    )
    print("[INFO] create new resnet152 model.")
    return model