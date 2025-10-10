# Creating Animated GIF from MISO Heatmaps

## Option 1: Manual Screenshot Method (Recommended)

### Step 1: Take Screenshots
1. Open each heatmap in your browser:
   - `output/heatmaps/miso_full_heatmap_12h.html`
   - `output/heatmaps/miso_full_heatmap_24h.html`
   - `output/heatmaps/miso_full_heatmap_36h.html`
   - `output/heatmaps/miso_full_heatmap_48h.html`

2. For each map:
   - Zoom to show the full MISO territory
   - Take a screenshot (Cmd+Shift+4 on Mac, Windows+Shift+S on Windows)
   - Save as: `screenshot_12h.png`, `screenshot_24h.png`, etc.

### Step 2: Create GIF Online
1. Go to an online GIF maker like:
   - https://ezgif.com/maker
   - https://gifmaker.me/
   - https://www.canva.com/create/animated-gifs/

2. Upload your 4 screenshots in order (12h → 24h → 36h → 48h)
3. Set duration: 1.5-2 seconds per frame
4. Set to loop continuously
5. Download your animated GIF

## Option 2: Using Command Line Tools

If you have ImageMagick installed:
```bash
# Convert screenshots to GIF
convert -delay 150 -loop 0 screenshot_*.png miso_animation.gif
```

## Option 3: Using Python (if you have the screenshots)
```python
from PIL import Image
import glob

# Load images
images = []
for filename in sorted(glob.glob('screenshot_*.png')):
    images.append(Image.open(filename))

# Save as GIF
images[0].save('miso_animation.gif', save_all=True, 
               append_images=images[1:], duration=1500, loop=0)
```

## Tips for Best Results
- Use consistent zoom level for all screenshots
- Ensure all maps show the same geographic area
- Keep file size reasonable for presentations (<10MB)
- Test the animation speed - 1.5-2 seconds per frame works well
