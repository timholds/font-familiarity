def generate_training_data(font_path, text_samples):
    """
    Generate training images for a single font
    Returns: tensor of shape (n_samples, 1, 64, 512)
    """
    images = []
    transform = transforms.Compose([
        transforms.Resize((64, 512)),
        transforms.ToTensor()
    ])
    
    # Create blank image and render text
    font = ImageFont.truetype(font_path, 48)
    for text in text_samples:
        img = Image.new('L', (512, 64), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), text, font=font, fill='black')
        images.append(transform(img))
    
    return torch.stack(images)