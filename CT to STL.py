import torch
import torch.nn as nn
import vtk
import glob
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from vtk.util import numpy_support
from sklearn.model_selection import train_test_split

class VascularDataset(Dataset):
    def __init__(self, list_of_vti_files):
        self.list_of_vti_files = list_of_vti_files

    def __len__(self):
        return len(self.list_of_vti_files)

    def __getitem__(self, idx):
        current_filepath = self.list_of_vti_files[idx]

        reader = vtk.vtkXMLImageDataReader()
        reader.SetFileName(current_filepath)
        reader.Update()

        dims = reader.GetOutput().GetDimensions()

        image_data = reader.GetOutput().GetPointData().GetScalars()
        image_numpy = numpy_support.vtk_to_numpy(image_data)
        image_numpy = image_numpy.reshape(dims[2], dims[1], dims[0])

        threshold_filter = vtk.vtkImageThreshold()
        threshold_filter.SetInputConnection(reader.GetOutputPort())
        threshold_filter.ThresholdBetween(150, 450)
        threshold_filter.SetInValue(1)
        threshold_filter.SetOutValue(0)
        threshold_filter.Update()

        raw_data = threshold_filter.GetOutput().GetPointData().GetScalars()
        mask_numpy = numpy_support.vtk_to_numpy(raw_data)
        mask_numpy = mask_numpy.reshape(dims[2], dims[1], dims[0])

        image_tensor = torch.from_numpy(image_numpy).float().unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_numpy).float().unsqueeze(0)

        image_tensor = (image_tensor - image_tensor.min()) / (image_tensor.max() - image_tensor.min() + 1e-8)

        image_tensor = image_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.unsqueeze(0)

        image_tensor = torch.nn.functional.interpolate(image_tensor, size=(128, 128, 128), mode='trilinear', align_corners=False)
        mask_tensor = torch.nn.functional.interpolate(mask_tensor, size=(128, 128, 128), mode='nearest')

        image_tensor = image_tensor.squeeze(0)
        mask_tensor = mask_tensor.squeeze(0)

        return image_tensor, mask_tensor
    
class UNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.enc1 = self.conv_block(1, 16)
        self.enc2 = self.conv_block(16, 32)
        self.enc3 = self.conv_block(32, 64)
        self.pool = nn.MaxPool3d(2)

        self.bottleneck = self.conv_block(64, 128)

        self.up3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(128, 64)
        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(64, 32)
        self.up1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(32, 16)

        self.final = nn.Conv3d(16, 1, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.final(d1))


folder_path = r"<vti file path>\*.vti"
file_list = glob.glob(folder_path)

train_files, val_files = train_test_split(file_list, test_size=0.2, random_state=42)

train_dataset = VascularDataset(train_files)
val_dataset = VascularDataset(val_files)

train_loader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)
val_loader   = DataLoader(dataset=val_dataset,   batch_size=1, shuffle=False)

model = UNet3D()


model_path = r"<.pt file path>\aorta_segmentation_model.pt"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))

def dice_loss(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def combined_loss(pred, target):
    return nn.BCELoss()(pred, target) + dice_loss(pred, target)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


for epoch in range(3):
    for images, masks in train_loader:
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = combined_loss(outputs, masks)
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch} - Loss: {loss.item()}")

torch.save(model.state_dict(), r"<.pt file path>/aorta_segmentation_model.pt")


test_image, test_mask = val_dataset[0]
test_image = test_image.unsqueeze(0)

model.eval()
with torch.no_grad():
    prediction = model(test_image)

prediction = prediction.squeeze(0).squeeze(0).numpy()


flat_prediction = prediction.ravel()
vtk_array = numpy_support.numpy_to_vtk(flat_prediction, deep=True, array_type=vtk.VTK_FLOAT)

depth, height, width = 128, 128, 128

predicton_vtk = vtk.vtkImageData()
predicton_vtk.SetDimensions(width, height, depth)
predicton_vtk.GetPointData().SetScalars(vtk_array)


marching_cubes = vtk.vtkMarchingCubes()
marching_cubes.SetInputData(predicton_vtk)
marching_cubes.SetValue(0, 0.5)
marching_cubes.Update()

connectivity = vtk.vtkPolyDataConnectivityFilter()
connectivity.SetInputConnection(marching_cubes.GetOutputPort())
connectivity.SetExtractionModeToLargestRegion()
connectivity.Update()

smoother = vtk.vtkSmoothPolyDataFilter()
smoother.SetInputConnection(connectivity.GetOutputPort())
smoother.SetNumberOfIterations(50)
smoother.SetRelaxationFactor(0.1)
smoother.Update()

normals = vtk.vtkPolyDataNormals()
normals.SetInputConnection(smoother.GetOutputPort())
normals.Update()


writer = vtk.vtkSTLWriter()
writer.SetFileName(r"<.stl export file path>\predicted_aorta.stl")
writer.SetInputConnection(normals.GetOutputPort())
writer.Write()

print("STL created")