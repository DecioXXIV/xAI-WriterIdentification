import PIL

def create_image_grid(
        crop_size:int, 
        overlap:int, 
        img:PIL.Image):
    """
    Args:
        crop_size (int): size of the (square) crop
        overlap (int): overlap (n_pixels) between adjacent crops
        img (PIL.Image): image to be cropped
    
    Returns:
        grid_dict (dict): dictionary with keys as the coordinates of the grid and values as the coordinates of the crop
        num_cols (int): number of columns in the grid
        num_rows (int): number of rows in the grid
    """
    img_w, img_h = img.size
    num_cols = (img_w - overlap)//(crop_size - overlap)
    num_rows = (img_h - overlap)//(crop_size - overlap)

    grid_w = (num_cols - 1)*(crop_size - overlap) + crop_size
    grid_h = (num_rows - 1)*(crop_size - overlap) + crop_size

    UL_x = int((img_w - grid_w)/2)
    UL_y = int((img_h - grid_h)/2)

    grid_dict = {}

    for i in range(num_rows):
        for j in range(num_cols):
            UL_x_grid = UL_x + j*(crop_size - overlap)
            UL_y_grid = UL_y + i*(crop_size - overlap)
            BR_x_grid = UL_x_grid + crop_size
            BR_y_grid = UL_y_grid + crop_size
            grid_dict[f'{i}_{j}'] = (UL_x_grid, UL_y_grid, BR_x_grid, BR_y_grid)
    
    return grid_dict, num_cols, num_rows