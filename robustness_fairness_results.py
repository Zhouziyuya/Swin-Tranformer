

# robustness

results = {  # CheXpert
        "CARZero": {"no_aug": 92.38, "metal": 91.64, "brightness": 91.59, "gaussian": 91.54},
        "RadZero": {"no_aug": 90.16, "metal": 90.13, "brightness": 90.1, "gaussian": 90.14},
        "DeViDe": {"no_aug": 89.87, "metal": 89.18, "brightness": 89.13, "gaussian": 89.02},
        "KAD": {"no_aug": 89.23, "metal": 83.86, "brightness": 84.81, "gaussian": 84.10},
        "MAVL": {"no_aug": 90.13, "metal": 88.61, "brightness": 89.8, "gaussian": 88.96},
        "MedKLIP": {"no_aug": 90.06, "metal": 88.16, "brightness": 88.18, "gaussian": 88.39},
        "Ark+": {"no_aug": 89.67, "metal": 89.87, "brightness": 89.83, "gaussian": 89.6},
        "FoundationX": {"no_aug": 89.55, "metal": 85.92, "brightness": 87.36, "gaussian": 87.14},
        "RAD-DINO": {"no_aug": 87.87, "metal": 87.4, "brightness": 87.77, "gaussian": 87.76},
        "EVA-X": {"no_aug": 88.02, "metal": 81.67, "brightness": 88.19, "gaussian": 88.04},
        "CheXWorld": {"no_aug": 85.84, "metal": 81.94, "brightness": 84.5, "gaussian": 83.89},
        "Lamps": {"no_aug": 82.68, "metal": 81.74, "brightness": 82.25, "gaussian": 81.84},
        "Adam-v2": {"no_aug": 80.81, "metal": 71.2, "brightness": 74.06, "gaussian": 73.53},
    }
    
    


result2 = {  # ChestXray-14
        "CARZero": {"no_aug": 77.67, "metal": 76.58, "brightness": 77.44, "gaussian": 76.63},
        "RadZero": {"no_aug": 75.92, "metal": 75.74, "brightness": 75.82, "gaussian": 75.23},
        "DeViDe": {"no_aug": 77.61, "metal": 75.42, "brightness": 76.71, "gaussian": 75.63},
        "KAD": {"no_aug": 76.99, "metal": 74.81, "brightness": 75.96, "gaussian": 74.86},
        "MAVL": {"no_aug": 73.43, "metal": 66.91, "brightness": 71.10, "gaussian": 67.19},
        "MedKLIP": {"no_aug": 72.68, "metal": 70.95, "brightness": 72.51, "gaussian": 71.09}, # need update
        "Ark+": {"no_aug": 84.42, "metal": 84.29, "brightness": 84.41, "gaussian": 83.28},
        "FoundationX": {"no_aug": 83.42, "metal": 81.21, "brightness": 82.49, "gaussian": 79.95},
        "RAD-DINO": {"no_aug": 79.98, "metal": 79.3, "brightness": 79.97, "gaussian": 79.81},
        "EVA-X": {"no_aug": 79.8, "metal": 74.75, "brightness": 79.63, "gaussian": 78.47},
        "CheXWorld": {"no_aug": 78.26, "metal": 73.3, "brightness": 78.09, "gaussian": 76.08},
        "Lamps": {"no_aug": 72.89, "metal": 72.65, "brightness": 72.85, "gaussian": 71.68},
        "Adam-v2": {"no_aug": 72.88, "metal": 66.66, "brightness": 68.31, "gaussian": 64.66},
    }


results3 = {  # RSNA
        "CARZero": {"no_aug": 77.74, "metal": 73.92, "brightness": 70.72, "gaussian": 70.05},   
        "RadZero": {"no_aug": 85.42, "metal": 84.77, "brightness": 85.02, "gaussian": 84.4},
        "DeViDe": {"no_aug": 88.58, "metal": 81.33, "brightness": 87.16, "gaussian": 78.63}, 
        "KAD": {"no_aug": 85.32, "metal": 82.71, "brightness": 86.20, "gaussian": 81.50},
        "MAVL": {"no_aug": 90.69, "metal": 91.03, "brightness": 91.23, "gaussian": 91.38},   
        "MedKLIP": {"no_aug": 89.06, "metal": 85.58, "brightness": 89.76, "gaussian": 86.56},    
        "Ark+": {"no_aug": 88.55, "metal": 88.47, "brightness": 88.51, "gaussian": 88.23},
        "FoundationX": {"no_aug": 87.05, "metal": 85.92, "brightness": 86.5, "gaussian": 86.04},
        "RAD-DINO": {"no_aug": 85.47, "metal": 85.46, "brightness": 85.48, "gaussian": 85.46},
        "EVA-X": {"no_aug": 85.62, "metal": 83.61, "brightness": 85.62, "gaussian": 85.47},
        "CheXWorld": {"no_aug": 84.29, "metal": 82.67, "brightness": 84.36, "gaussian": 84.14},
        "Lamps": {"no_aug": 82.5, "metal": 82.38, "brightness": 82.52, "gaussian": 81.45},
        "Adam-v2": {"no_aug": 81.83, "metal": 77.23, "brightness": 76.84, "gaussian": 75.3},
    }

results4 = {  # CovidQuEx
        "CARZero": {"no_aug": 83.76, "metal": 62.29, "brightness": 81.83, "gaussian": 77.55},       
        "RadZero": {"no_aug": 86.57, "metal": 79.01, "brightness": 86.28, "gaussian": 86.57}, 
        "DeViDe": {"no_aug": 87.03, "metal": 71.22, "brightness": 88.16, "gaussian": 83.01},    
        "KAD": {"no_aug": 87.81, "metal": 65.85, "brightness": 87.40, "gaussian": 83.93},   
        "MAVL": {"no_aug": 87.39, "metal": 84.47, "brightness": 86.92, "gaussian": 87.61},      
        "MedKLIP": {"no_aug": 82.79, "metal": 63.21, "brightness": 84.66, "gaussian": 80.19},       
        "Ark+": {"no_aug": 99.05, "metal": 98.85, "brightness": 99, "gaussian": 94.86},
        "FoundationX": {"no_aug": 97.27, "metal": 96.57, "brightness": 97.27, "gaussian": 92.74},
        "RAD-DINO": {"no_aug": 98.95, "metal": 98.76, "brightness": 98.95, "gaussian": 98.72},
        "EVA-X": {"no_aug": 98.19, "metal": 94.95, "brightness": 98.19, "gaussian": 96.46},
        "CheXWorld": {"no_aug": 97.67, "metal": 96.26, "brightness": 97.68, "gaussian": 97.39},
        "Lamps": {"no_aug": 96.42, "metal": 96.23, "brightness": 96.25, "gaussian": 87.92},
        "Adam-v2": {"no_aug": 96.48, "metal": 96.71, "brightness": 96.48, "gaussian": 91.7},
    }



# fairness
# CheXpert - DEOdds (Lower is better)
chexpert_deodds = {
    "KAD": {"sex": 9.03, "age": 26.74, "race": 48.62},
    "DeViDe": {"sex": 11.99, "age": 23.72, "race": 31.87},
    "CARZero": {"sex": 14.21, "age": 31.20, "race": 35.01},
    "RadZero": {"sex": 13.28, "age": 31.25, "race": 25.98},
    "MAVL": {"sex": 15.07, "age": 26.45, "race": 33.54},
    "MedKLIP": {"sex": 14.51, "age": 39.41, "race": 40.91},
    "Ark+": {"sex": 11.16, "age": 28.98, "race": 22.41},
    "FoundationX": {"sex": 10.99, "age": 28.41, "race": 25.90},
    "RAD-DINO": {"sex": 9.06, "age": 20.01, "race": 26.25},
    "EVA-X": {"sex": 5.57, "age": 16.68, "race": 18.44},
    "Lamps": {"sex": 5.47, "age": 15.66, "race": 19.16},
    "Adam-v2": {"sex": 6.29, "age": 11.51, "race": 20.12},
    "CheXWorld": {"sex": 4.80, "age": 20.70, "race": 20.11},
}

# CheXpert - Max-Min Gap (Lower is better)
chexpert_max_min_gap = {
    "KAD": {"sex": 3.32, "age": 10.50, "race": 11.30},
    "DeViDe": {"sex": 2.64, "age": 9.72, "race": 12.36},
    "CARZero": {"sex": 3.00, "age": 5.13, "race": 10.13},
    "RadZero": {"sex": 4.32, "age": 6.32, "race": 18.09},
    "MAVL": {"sex": 3.39, "age": 5.21, "race": 32.04},
    "MedKLIP": {"sex": 5.32, "age": 5.67, "race": 13.80},
    "Ark+": {"sex": 1.05, "age": 8.95, "race": 8.44},
    "FoundationX": {"sex": 1.45, "age": 8.27, "race": 9.70},
    "RAD-DINO": {"sex": 1.48, "age": 13.09, "race": 2.13},
    "EVA-X": {"sex": 1.71, "age": 9.16, "race": 11.07},
    "Lamps": {"sex": 2.67, "age": 7.38, "race": 20.46},
    "Adam-v2": {"sex": 1.22, "age": 8.42, "race": 4.20},
    "CheXWorld": {"sex": 0.71, "age": 9.88, "race": 20.11},
}


# ChestX-ray14 - DEOdds (Lower is better)
chestxray14_deodds = {
    "KAD": {"gender": 6.10, "age": 29.47},
    "DeViDe": {"gender": 6.56, "age": 26.86},
    "CARZero": {"gender": 5.29, "age": 35.13},
    "RadZero": {"gender": 5.30, "age": 34.94},
    "MAVL": {"gender": 4.49, "age": 31.90},
    "MedKLIP": {"gender": 5.14, "age": 28.88},
    "Ark+": {"gender": 2.12, "age": 12.28},
    "FoundationX": {"gender": 1.96, "age": 12.28},
    "RAD-DINO": {"gender": 2.44, "age": 11.86},
    "EVA-X": {"gender": 2.50, "age": 6.47},
    "Lamps": {"gender": 1.10, "age": 1.65},
    "Adam-v2": {"gender": 0.45, "age": 2.27},
    "CheXWorld": {"gender": 2.64, "age": 0.50},
}

# ChestX-ray14 - Max-Min Gap (Lower is better)
chestxray14_max_min_gap = {
    "KAD": {"gender": 1.11, "age": 5.59},
    "DeViDe": {"gender": 1.20, "age": 7.39},
    "CARZero": {"gender": 0.85, "age": 6.50},
    "RadZero": {"gender": 1.06, "age": 6.77},
    "MAVL": {"gender": 1.17, "age": 5.07},
    "MedKLIP": {"gender": 0.64, "age": 6.46},
    "Ark+": {"gender": 0.43, "age": 4.41},
    "FoundationX": {"gender": 0.78, "age": 4.67},
    "RAD-DINO": {"gender": 0.16, "age": 7.22},
    "EVA-X": {"gender": 0.50, "age": 4.24},
    "Lamps": {"gender": 0.16, "age": 11.87},
    "Adam-v2": {"gender": 0.06, "age": 7.94},
    "CheXWorld": {"gender": 0.48, "age": 8.39},
}
