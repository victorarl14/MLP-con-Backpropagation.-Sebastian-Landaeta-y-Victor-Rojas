from PIL import Image, ImageDraw

# Crear una imagen en blanco (fondo negro)
img = Image.new('L', (28, 28), color=0)
draw = ImageDraw.Draw(img)

# Dibujar un dígito (por ejemplo, el número "3")
draw.line([(5, 5), (20, 5)], fill=255, width=2)  # Línea superior
draw.line([(5, 10), (20, 10)], fill=255, width=2)  # Línea media
draw.line([(5, 15), (20, 15)], fill=255, width=2)  # Línea inferior
draw.line([(20, 5), (20, 15)], fill=255, width=2)  # Línea derecha

# Guardar la imagen
img.save('digito.png')