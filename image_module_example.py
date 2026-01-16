from PIL import Image

#see outputs from example in example_images
im =  Image.open("example_images/floral_pattern.jpg") #opening file as Image object
print(im.mode) #rgb by defualt
print(im.getbands()) #('R', 'G', 'B')

#conversion to grayscale
gray_im = im.convert("L")
gray_im.save("example_images/grayscale_floral_pattern.jpg")

#split function
red, green, blue = im.split() #red is a grayscale image derived from the r values in im
zeroed_band = red.point(lambda _: 0) #all 0 band, same dimensions as red
red_merge = Image.merge("RGB", (red, zeroed_band, zeroed_band)) #take red to RGB, using zeros for G, B
red_merge.save("example_images/red_only_floral_pattern.jpg")