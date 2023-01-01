#Renaming RectangleRegion to RectangleWindow
class RectangleWindow:
    # x coordinate of the start position of the region
    # y coordinate of the start position of the region
    # width of the rectangle region
    # height of the rectangle region
                    
    def __init__ ( self , x_start_position_of_the_window , y_start_position_of_the_window , window_width , window_height ):
        
        self.x_start_position_of_the_window = x_start_position_of_the_window
        self.y_start_position_of_the_window = y_start_position_of_the_window
        
        self.window_width = window_width
        self.window_height = window_height

    # used to get the sum of pixels in a specific region 
    # inside an image using only 4 points of the integral image instead of all points in the region in original image
    # using integer is to make sure that all values are integers
        
    def get_window_sum(self, integral_image, scale = 1.0):
        
        x_window_start_top_left = int( self.x_start_position_of_the_window * scale )
        y_window_start_top_left = int( self.y_start_position_of_the_window * scale )
        
        bottom_right_x = x_window_start_top_left + int( self.window_width * scale ) - 1
        bottom_right_y = y_window_start_top_left + int( self.window_height * scale ) - 1
        
        window_sum = 0
        window_sum = int(integral_image[bottom_right_x, bottom_right_y])
        
        if x_window_start_top_left > 0:
            window_sum -= int( integral_image[ x_window_start_top_left-1 , bottom_right_y ])
        
        if y_window_start_top_left > 0:
            window_sum -= int( integral_image[ bottom_right_x, y_window_start_top_left-1 ])
        
        if x_window_start_top_left > 0 and y_window_start_top_left > 0:
            window_sum += int( integral_image[ x_window_start_top_left - 1 , y_window_start_top_left - 1 ])
        
        return window_sum

