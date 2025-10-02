from PIL import Image, ImageDraw, ImageFont

def write_text_on_image(image_path, text, font_path="arial.ttf",
                        font_size=20, text_color=(0, 0, 0), background_color=(0, 0, 0)):
    image = Image.open(image_path)
    width, height = image.size
    padding_top = 10

    new_height = height + padding_top
    new_image = Image.new("RGB", (width, new_height), (255, 255, 255))

    new_image.paste(image, (0, padding_top))
    image = new_image

    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=font_size)

    text_width, text_height = draw.textbbox((0, 0), text=text, font=font)[2:]

    rect_height = text_height + padding_top
    rect_width = image.width

    draw.rectangle([(0, 0), (rect_width, rect_height)], fill=background_color)

    # Define text position (top center over the black rectangle)
    text_x = (image.width - text_width) / 2
    text_y = (rect_height - text_height) / 2

    # Add text to image
    draw.text((text_x, text_y), text, fill=text_color, font=font)
    # return the modified image
    return image