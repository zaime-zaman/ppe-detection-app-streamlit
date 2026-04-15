"""
OpenCV compatibility layer using PIL for Streamlit Cloud deployment.
Replaces cv2 with PIL-based equivalents.
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np

# Font constants (mapped to PIL)
FONT_HERSHEY_SIMPLEX = "default"
COLOR_BGR2RGB = "BGR2RGB"
COLOR_RGB2BGR = "RGB2BGR"
INTER_AREA = "LANCZOS"

def cvtColor(img, code):
    """Convert between color spaces (BGR ↔ RGB)"""
    if code == COLOR_BGR2RGB:
        # OpenCV uses BGR, PIL uses RGB - swap channels
        return img[..., ::-1]  # Reverse last axis (B,G,R -> R,G,B)
    elif code == COLOR_RGB2BGR:
        return img[..., ::-1]
    return img

def resize(img, new_size, interpolation=None):
    """Resize image to new size (width, height)"""
    pil_img = Image.fromarray(img.astype('uint8'))
    # new_size is (width, height) for cv2, same for PIL
    pil_img = pil_img.resize(new_size, Image.LANCZOS)
    return np.array(pil_img)

def rectangle(img, pt1, pt2, color, thickness):
    """Draw rectangle on image (in-place)"""
    pil_img = Image.fromarray(img.astype('uint8'))
    draw = ImageDraw.Draw(pil_img)
    # Convert BGR to RGB for PIL
    color_rgb = (color[2], color[1], color[0]) if len(color) == 3 else color
    draw.rectangle([pt1, pt2], outline=color_rgb, width=thickness)
    np.copyto(img, np.array(pil_img))

def putText(img, text, org, fontFace, fontScale, color, thickness):
    """Put text on image (in-place)"""
    pil_img = Image.fromarray(img.astype('uint8'))
    draw = ImageDraw.Draw(pil_img)
    
    # Convert BGR to RGB for PIL
    color_rgb = (color[2], color[1], color[0]) if len(color) == 3 else color
    
    # Approximate font size from OpenCV's fontScale
    # OpenCV default is ~0.5-1.0 for 640x480, we'll use 12-16 pixels
    font_size = max(8, int(fontScale * 20))
    
    try:
        # Try to use a truetype font if available
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        # Fall back to default PIL font
        font = ImageFont.load_default()
    
    # org is (x, y) - text position
    draw.text(org, text, fill=color_rgb, font=font)
    np.copyto(img, np.array(pil_img))
