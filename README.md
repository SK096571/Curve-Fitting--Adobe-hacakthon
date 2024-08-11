# Curve-Fitting--Adobe-hacakthon
Tech Trivia
### 1. Data Reading and Preprocessing
**1.1 Reading and Preprocessing Data**
Load polyline data from CSV or image formats, and preprocess using OpenCV and NumPy for
normalization and cleaning.
```python
import numpy as np
import pandas as pd
import cv2
def read_csv(csv_path):
df = pd.read_csv(csv_path, header=None)
path_XYs = []
for i in df[0].unique():
path_df = df[df[0] == i]
segments = path_df[1].unique()
XYs = [path_df[path_df[1] == j].iloc[:, 2:].values for j in segments]
path_XYs.append(XYs)
return path_XYs
def preprocess_image(image_path):
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL,
cv2.CHAIN_APPROX_SIMPLE)
return [contour.squeeze() for contour in contours if contour.size > 0]
```
### 2. Curve Detection and Regularization
**2.1 Advanced Curve Detection**
Use OpenCV for detecting simple curves and integrate scikit-learn and scipy for fitting complex
curves.
```python
from sklearn.linear_model import RANSACRegressor
from scipy.optimize import curve_fit
from scipy.interpolate import splprep, splev
def detect_lines(XYs):
lines = []
for segment in XYs:
x = segment[:, 0].reshape(-1, 1)
y = segment[:, 1]
model = RANSACRegressor()
model.fit(x, y)
inliers = model.inlier_mask_
lines.append({
'slope': model.estimator_.coef_[0],
'intercept': model.estimator_.intercept_,
'inliers': segment[inliers]
})
return lines
def fit_circle(XYs):
def circle_residuals(params, x, y):
xc, yc, R = params
return np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - R
x = XYs[:, 0]
y = XYs[:, 1]
x0, y0 = np.mean(x), np.mean(y)
R0 = np.mean(np.sqrt((x - x0) ** 2 + (y - y0) ** 2))
params_initial = [x0, y0, R0]
params_opt, _ = curve_fit(lambda p, x, y: circle_residuals(p, x, y), params_initial, x=x, y=y)
return params_opt
def fit_ellipse(XYs):
if len(XYs) >= 5:
ellipse = cv2.fitEllipse(XYs)
return ellipse
return None
def fit_polynomial(XYs, degree=5):
x = XYs[:, 0].reshape(-1, 1)
y = XYs[:, 1]
poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(x)
model = LinearRegression().fit(X_poly, y)
return model, poly
def fit_spline(XYs, s=0):
x = XYs[:, 0]
y = XYs[:, 1]
tck, _ = splprep([x, y], s=s)
return tck
```
### 3. Symmetry Detection
**3.1 Using Deep Learning for Symmetry**
Leverage deep learning models to detect symmetrical patterns and regularities.
```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
def detect_symmetry(image_path):
model = models.resnet18(pretrained=True)
model.eval()
preprocess = transforms.Compose([
transforms.Resize(256),
transforms.CenterCrop(224),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img = Image.open(image_path)
img_tensor = preprocess(img).unsqueeze(0)
with torch.no_grad():
outputs = model(img_tensor)
return outputs
```
**3.2 Reflection and Rotational Symmetry**
Analyze symmetry by comparing original and transformed curves.
```python
from shapely.geometry import LineString
import numpy as np
def check_reflection_symmetry(XYs):
centroid = np.mean(XYs, axis=0)
line = LineString([(np.min(XYs[:, 0]), centroid[1]), (np.max(XYs[:, 0]), centroid[1])])
reflected = np.copy(XYs)
reflected[:, 0] = 2 * centroid[0] - reflected[:, 0]
return np.allclose(np.sort(XYs[:, 0]), np.sort(reflected[:, 0]))
def check_rotational_symmetry(XYs, num_rotations=360):
centroid = np.mean(XYs, axis=0)
for angle in np.linspace(0, 360, num_rotations):
rotated = rotate(XYs, angle, origin=centroid)
if np.allclose(XYs, rotated):
return angle
return None
```
### 4. Curve Completion
**4.1 Using Deep Learning for Curve Completion**
Leverage pre-trained deep learning models for curve completion and inpainting.
```python
import tensorflow as tf
def complete_curve_with_deep_learning(model_path, XYs):
model = tf.keras.models.load_model(model_path)
XYs_normalized = (XYs - np.mean(XYs, axis=0)) / np.std(XYs, axis=0)
XYs_pred = model.predict(XYs_normalized)
return XYs_pred
```
**4.2 Interpolation Techniques**
Use interpolation for filling gaps in incomplete curves.
```python
from scipy.interpolate import interp1d
def interpolate_missing_points(XYs):
x = XYs[:, 0]
y = XYs[:, 1]
t = np.arange(len(x))
t_new = np.linspace(0, len(x)-1, len(x))
interp_x = interp1d(t, x, kind='cubic')
interp_y = interp1d(t, y, kind='cubic')
x_interp = interp_x(t_new)
y_interp = interp_y(t_new)
return np.column_stack((x_interp, y_interp))
```
### 5. Bézier Curve Fitting
**5.1 Robust Fitting of Bézier Curves**
Apply optimization techniques for cubic Bézier curve fitting.
```python
from scipy.optimize import minimize
def bezier_curve(t, p0, p1, p2, p3):
return (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3
def bezier_residuals(params, XYs):
p0, p1, p2, p3 = np.array(params).reshape(4, 2)
t = np.linspace(0, 1, len(XYs))
curve = np.array([bezier_curve(ti, p0, p1, p2, p3) for ti in t])
return np.sum((curve - XYs) ** 2)
def fit_bezier_curve(XYs):
initial_guess = np.concatenate([XYs[0], XYs[len(XYs) // 2], XYs[-1]])
result = minimize(bezier_residuals, initial_guess, args=(XYs,), method='L-BFGS-B')
if not result.success:
raise ValueError("Optimization failed to converge.")
p0, p1, p2, p3 = np.array(result.x).reshape(4, 2)
return p0, p1, p2, p3
```
### 6. Integration and Visualization
**6.1 Comprehensive Pipeline**
Combine all components into a single pipeline for efficient processing and visualization.
```python
import matplotlib.pyplot as plt
from shapely.affinity import rotate
def visualize_curves(original_XYs, completed_XYs, bezier_params):
plt.figure(figsize=(10, 10))
plt.plot(original_XYs[:, 0], original_XYs[:, 1], 'b-', label='Original Curve')
plt.plot(completed_XYs[:, 0], completed_XYs[:, 1], 'r--', label='Completed Curve')
t = np.linspace(0, 1, 100)
bezier_curve = lambda t: (1 - t)**3 * bezier_params[0] + 3 * (1 - t)**2 * t * bezier_params[1] +
3 * (1 - t) * t**2 * bezier_params[2] + t**3 * bezier_params[3]
bezier_points = np.array([bezier_curve(ti) for ti in t])
plt.plot(bezier_points[:, 0], bezier_points[:, 1], 'g-.', label='Bézier Curve')
plt.legend()
plt.title('Curve Fitting and Completion')
plt.xlabel('
X')
plt.ylabel('Y')
plt.show()
def main_pipeline(image_path, model_path):
# Load and preprocess data
contours = preprocess_image(image_path)
for contour in contours:
# Regularize curves
lines = detect_lines(contour)
circle_params = fit_circle(contour)
ellipse_params = fit_ellipse(contour)
poly_model, poly = fit_polynomial(contour)
spline_tck = fit_spline(contour)
# Symmetry detection
symmetry = detect_symmetry(image_path)
rotational_symmetry = check_rotational_symmetry(contour)
# Curve completion
completed_curve = complete_curve_with_deep_learning(model_path, contour)
# Bézier fitting
bezier_params = fit_bezier_curve(contour)
# Visualization
visualize_curves(contour, completed_curve, bezier_params)
if __name__ == "__main__":
image_path = 'path/to/image.png'
model_path = 'path/to/deep_learning_model.h5'
main_pipeline(image_path, model_path)
