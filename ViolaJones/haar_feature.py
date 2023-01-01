class HaarFeature:

    def __init__(self, white_postive_regions, black_negative_regions):
        self.positive_regions = white_postive_regions  
        self.negative_regions = black_negative_regions  

    def get_haar_feature_value(self, integral_image, scale=1):
        """
        Compute the value of a rectangle feature(x,y,w,h) at the integral image
        each haar feature is divided into 2 symmetric regions: black and white,
        the sub of the black - white regions is the value of this haar feature
        """
        # to get the white region value
        white_positive_region_sum = sum([rectangle.get_window_sum(integral_image, scale) for rectangle in self.positive_regions])
        
        # to get the black region value
        black_negative_region_sum = sum([rectangle.get_window_sum(integral_image, scale) for rectangle in self.negative_regions])
        
        ## getting the feature value
        return black_negative_region_sum - white_positive_region_sum

