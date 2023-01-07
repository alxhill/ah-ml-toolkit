from toolkit.read_images import images_from_dir, constrain_to_size, imgs_to_tensors

if __name__ == "__main__":
    inputs = images_from_dir("../data/raw/living_room")
    resized = constrain_to_size(inputs, (512, 512))
    tensors = imgs_to_tensors(resized)
