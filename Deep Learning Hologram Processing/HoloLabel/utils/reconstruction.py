import torch
import math
import numpy as np

from tqdm import tqdm

# Need to add shift peak


def adjustPeak(intensityProj, adjust_peak, substraction_factor):
    bin_list, _ = np.histogram(intensityProj, bins=list(range(256)))
    current_peak = list(range(256))[np.argmax(bin_list)]
    intensityProj = intensityProj.astype(float)
    if current_peak > 0:
        bottom_divide = float(current_peak - substraction_factor)

        intensityProj = (
            (intensityProj - substraction_factor) * adjust_peak / (bottom_divide + 1)
        )
        intensityProj[intensityProj > 255] = 255
    intensityProj = intensityProj.astype(np.uint8)

    return intensityProj


class Reconstruction(object):
    def __init__(
        self,
        resolution,
        wavelength,
        z_start,
        z_step,
        num_planes,
        im_x,
        im_y,
        shift_mean,
        shift_value,
        padding=32,
    ):
        # Parse parameters
        im_x = im_x + padding * 2
        im_y = im_y + padding * 2
        if torch.cuda.is_available():
            cuda = torch.device("cuda")
        else:
            cuda = torch.device("cpu")
        if resolution < 0.5:
            ResolutionScaleFactor = 0.5 / resolution
            resolution = ResolutionScaleFactor * resolution
            z_start = ResolutionScaleFactor * ResolutionScaleFactor * z_start
            z_step = ResolutionScaleFactor * ResolutionScaleFactor * z_step
        self.resolution = torch.tensor(resolution, device=cuda)
        self.padding = padding
        self.wavelength = torch.tensor(wavelength, device=cuda)
        self.z_start = z_start
        self.z_step = z_step
        self.num_planes = num_planes
        self.im_x = torch.tensor(im_x, device=cuda)
        self.im_y = torch.tensor(im_y, device=cuda)
        self.shift_mean = shift_mean
        self.shift_value = shift_value
        # list of z values to reconstruct to
        self.z_end = self.num_planes * self.z_step + self.z_start        

        self.z_list = np.arange(self.z_start, self.z_end, self.z_step)

        # Calculate image position values
        x = torch.arange(self.im_x, device=cuda)
        y = torch.arange(self.im_y, device=cuda)
        xx, yy = torch.meshgrid(x, y)
        xx = torch.t(xx)
        yy = torch.t(yy)
        fx = xx / self.im_x - 0.5
        fy = yy / self.im_y - 0.5
        f2 = torch.multiply(fx, fx) + torch.multiply(fy, fy)
        self.f2_new = torch.fft.fftshift(f2)

        sqrt_input = 1 - f2 * torch.square(self.wavelength / self.resolution)
        sqrt_input = torch.clip(sqrt_input, min=0)
        self.H = -2 * math.pi * 1j * torch.sqrt(sqrt_input) / self.wavelength
        self.H = torch.fft.fftshift(self.H)
        self.temp = (self.wavelength / self.resolution) * (
            self.wavelength / self.resolution
        )
        self.f2_new = self.temp * self.f2_new
        self.f2_new = torch.sqrt(1 - self.f2_new)
    
    def rec_3D(self, im_gpu):
        # Assuming im_gpu is a torch tensor on the GPU
        im_new = torch.full((im_gpu.shape[0] + self.padding * 2, im_gpu.shape[1] + self.padding * 2), torch.median(im_gpu).item(), device=im_gpu.device)
        im_new[self.padding: self.padding + im_gpu.shape[0], self.padding: self.padding + im_gpu.shape[1]] = im_gpu

        Fholo = torch.fft.fft2(im_new)

        rec3 = []
        for idx, z in enumerate(self.z_list):
            Hz = torch.exp(-2 * math.pi * 1j * z * self.f2_new / self.wavelength).to(im_gpu.device)
            phase_temp = torch.exp(2 * math.pi * 1j * z / self.wavelength).to(im_gpu.device)
            Fplane = torch.multiply(Fholo, Hz)
            rec_temp = torch.multiply(torch.fft.ifft2(Fplane), phase_temp)

            if idx == 0:
                rec3 = rec_temp.detach().unsqueeze(-1)
            else:
                rec3 = torch.cat((rec3, rec_temp.detach().unsqueeze(-1)), dim=2)

        rec3 = rec3[self.padding: im_new.shape[0] - self.padding, self.padding: im_new.shape[1] - self.padding, :]
        return rec3
    
    def rec_3D_intensity(self, im):
        rec3 = self.rec_3D(im)
        
        # Intensity as magnitude squared of rec3
        intensity = torch.abs(rec3)**2

        # Normalize the intensity to the range [0, 1] (single pass min-max normalization)
        min_val = torch.min(intensity)
        max_val = torch.max(intensity)
        intensity = (intensity - min_val) / (max_val - min_val)
        
        mean_background_intensity = torch.mean(intensity)  # Mean over background pixels

        # Scale factor to shift background to desired value
        scale_factor = (self.shift_mean / 255) / mean_background_intensity
        intensity *= scale_factor

        # Clip the values to [0, 1] in one step
        intensity = torch.clamp(intensity, 0, 1)

        return intensity

    def rec_3D_intensity_bk(self, im):
        rec3 = self.rec_3D(im)
        intensity = torch.multiply(rec3, torch.conj(rec3))
        # intensity = np.array(torch.Tensor.cpu(intensity))
        intensity = np.array(intensity)
        intensity = np.real_if_close(intensity)
        intensity = (intensity - np.min(intensity)) / (
            np.max(intensity) - np.min(intensity)
        )
        return intensity

    def rec_3D_phase(self, im):
        rec3 = self.rec_3D(im)
        phase = rec3
        for i in range(rec3.shape[-1]):
            phase[:, :, i] = torch.angle(rec3[:, :, i])
        phase = torch.real(phase)
        phase = np.array(phase)
        return phase

    def rec_xy_intensity(self, im):
        intensity_3d = self.rec_3D_intensity(im)
        # xy_projection = np.max(intensity_3d, axis=-1)
        xy_projection = np.min(intensity_3d, axis=-1) * 255
        return xy_projection

    def rec_xz_intensity(self, im):
        intensity_3d = self.rec_3D_intensity(im)
        xz_projection = np.max(intensity_3d, axis=0)
        return xz_projection

    def rec_yz_intensity(self, im):
        intensity_3d = self.rec_3D_intensity(im)
        yz_projection = np.min(intensity_3d, axis=1)
        return yz_projection

    def rec_xy_phase(self, im):
        phase_3d = np.array(self.rec_3D_phase(im).cpu())
        # min or max, find the max or min location of phase location.
        xy_projection = np.max(phase_3d, axis=-1) / math.pi
        return xy_projection

    def getCentroidZ(self, im, xypos):
        intensity_3d = self.rec_3D_intensity(im)
        num_points = xypos.shape[0]
        z = np.zeros((num_points, 1))
        for n in range(num_points):
            x = int(round(xypos[n, 0]))
            y = int(round(xypos[n, 1]))
            vec = intensity_3d[y, x, :]
            plane = np.argmax(vec)
            z[n, 0] = plane / self.num_planes
        return z