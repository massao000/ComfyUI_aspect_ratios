from nodes import MAX_RESOLUTION
import torch
import os

class AspectRatio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "size": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                              "aspect_ratios": (s.aspect_ratios_label, ),
                              "standard": (['width', 'height'], ),
                              "swap_aspect_ratio": (['not_swap', 'swap'], ),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})
                            }
                }
    
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "run"
    CATEGORY = "Aspect Ratios"

    def aspect_ratios_file():
        my_path = os.path.dirname(__file__)
        requirements_path = os.path.join(my_path, "aspect_ratios.txt")

        if not os.path.exists(requirements_path):
            aspect_ratios = [
                "1:1, 1/1 # 1:1 ratio based on minimum dimension\n",
                "3:2, 3/2 # Set width based on 3:2 ratio to height\n",
                "4:3, 4/3 # Set width based on 4:3 ratio to height\n",
                "16:9, 16/9 # Set width based on 16:9 ratio to height",
            ]
            with open(requirements_path, "w", encoding="utf-8") as f:
                f.writelines(aspect_ratios)

        labels, values, comments = [], [], []
        with open(requirements_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            if line.startswith("#"):
                continue

            if ',' not in line:
                continue

            try:
                label, value = line.strip().split(",")
                comment = ""
                if "#" in value:
                    value, comment = value.split("#")
            except ValueError:
                print(f"skipping badly formatted line in aspect ratios file: {line}")
                continue

            labels.append(label)
            values.append(value)
            comments.append(comment)
            
        return labels, values
    
    aspect_ratios_label, aspect_ratios_num = aspect_ratios_file()

    def ar_size(self, is_width, aspect, size, is_swap):
        if is_swap:
            index = (self.aspect_ratios_num[self.aspect_ratios_label.index(aspect)])
            aspect_ratios = eval('/'.join(index.split('/')[::-1]))
        else:
            aspect_ratios = eval(self.aspect_ratios_num[self.aspect_ratios_label.index(aspect)])
        
        if is_width:
            result = size / aspect_ratios
        else:
            result = size * aspect_ratios
        
        return round(result)
    
    def run(self, size, aspect_ratios, standard, batch_size, swap_aspect_ratio):
        
        is_width = standard == "width"
        is_swap = swap_aspect_ratio == "swap"
        result_dimension = self.ar_size(is_width, aspect_ratios, size, is_swap)
        
        if is_width:    
            latent = torch.zeros([batch_size, 4, result_dimension // 8, size // 8])
        else:
            latent = torch.zeros([batch_size, 4, size // 8, result_dimension // 8])
            
        return ({"samples":latent}, )