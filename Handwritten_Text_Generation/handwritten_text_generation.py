from PIL import Image, ImageDraw, ImageFont

print("=" * 50)
print("HANDWRITTEN TEXT GENERATION")
print("=" * 50)

text = input("Enter text: ")

# Create image
img = Image.new("RGB", (1400, 600), "white")
draw = ImageDraw.Draw(img)

try:
    font = ImageFont.truetype("arial.ttf", 50)
except:
    font = ImageFont.load_default()

draw.text((50, 200), text, fill="black", font=font)

output_file = "handwritten_output.png"
img.save(output_file)

print("\nImage Generated Successfully!")
print(f"Saved as {output_file}")

# Open saved image
import os
os.startfile(output_file)