
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import boxcox
from scipy.stats import gaussian_kde

class data_extraction:
    @staticmethod
    def clean_exoplanet_data1(csv_path, output_path=None):
        
        # Read CSV file
        df = pd.read_csv(csv_path,comment = "#")

        # Remove rows missing required columns
        df = df.dropna(subset=["pl_orbeccen", "pl_orbsmax"])

        # Remove rows where eccentricity == 0 but error columns are missing
        # Check if eccentricity error columns exist
        
        df = df[~(df["pl_orbeccen"] == 0)]

        # Handle duplicates by planet name
        cleaned_rows = []
        for name, group in df.groupby("pl_name"):
            if len(group) == 1:
               cleaned_rows.append(group.iloc[0])
            else:
             # Compute mode of eccentricities for this planet
                mode_val = group["pl_orbeccen"].mode()
                if len(mode_val) == 0:
                # fallback if mode is empty
                    mode_val = group["pl_orbeccen"].median()
                else:
                    mode_val = mode_val.iloc[0]
            
            # Find datapoint closest to the mode
                closest_idx = (group["pl_orbeccen"] - mode_val).abs().idxmin()
                cleaned_rows.append(group.loc[closest_idx])

        # taking only the required columns
        final_data = []
        datapoints = ["pl_name","pl_orbsmax","pl_orbeccen"]
        for data in cleaned_rows:
            final_data.append([data.get(col, None) for col in datapoints])

        cleaned_df = pd.DataFrame(cleaned_rows)

        # Save cleaned data (optional)]
        if output_path:
            cleaned_df.to_csv(output_path, index=False)
            print(f"✅ Cleaned data saved to {output_path}")

        return final_data

    
    @staticmethod
    def clean_exoplanet_data2(csv_path, output_path=None):
        # Read CSV file while ignoring comments ---
        df = pd.read_csv(csv_path, comment="#")

        # Remove datapoints missing mass or density values ---
        df = df.dropna(subset=["pl_rade", "pl_bmasse"])

        # --- Step 5: Handle duplicates by planet name ---
        cleaned_rows = []
        for name, group in df.groupby("pl_name"):
           if len(group) == 1:
               cleaned_rows.append(group.iloc[0])
           else:
                mode_val = group["pl_bmasse"].mode()
                if len(mode_val) == 0:
                    mode_val = group["pl_bmasse"].median()
                else:
                    mode_val = mode_val.iloc[0]

            # Find row closest to mode
                closest_idx = (group["pl_bmasse"] - mode_val).abs().idxmin()
                cleaned_rows.append(group.loc[closest_idx])

        cleaned_df = pd.DataFrame(cleaned_rows)

        # taking only the required columns
        final_data = []
        datapoints = ["pl_name","pl_bmasse","pl_rade"]
        for data in cleaned_rows:
            f = [data.get(col, None) for col in datapoints]
            f[2] = np.float64(5514) * f[1]/f[2]**3
            final_data.append(f)
        
        if output_path:
            cleaned_df.to_csv(output_path, index=False)
            print(f"✅ Cleaned data saved to {output_path}")

        return final_data
    
    @staticmethod
    def clean_exoplanet_data3(csv_path, output_path=None):
        # Read CSV file while ignoring comments ---
        df = pd.read_csv(csv_path, comment="#")

        # Remove datapoints missing mass or density values ---
        df = df.dropna(subset=["pl_orbsmax"])

        # --- Step 5: Handle duplicates by planet name ---
        cleaned_rows = []
        for name, group in df.groupby("pl_name"):
           if len(group) == 1:
               cleaned_rows.append(group.iloc[0])
           else:
                mode_val = group["pl_orbsmax"].mode()
                if len(mode_val) == 0:
                    mode_val = group["pl_orbsmax"].median()
                else:
                    mode_val = mode_val.iloc[0]

            # Find row closest to mode
                closest_idx = (group["pl_orbsmax"] - mode_val).abs().idxmin()
                cleaned_rows.append(group.loc[closest_idx])

        cleaned_df = pd.DataFrame(cleaned_rows)

        # taking only the required columns
        final_data = []
        datapoints = ["pl_name","pl_orbsmax"]
        for data in cleaned_rows:
            f = [data.get(col, None) for col in datapoints]
            final_data.append(f)
        
        if output_path:
            cleaned_df.to_csv(output_path, index=False)
            print(f"✅ Cleaned data saved to {output_path}")

        return final_data
    
    @staticmethod
    def solar_system1(system_path):
        # read the solar system file
        df = pd.read_csv(system_path)

        final_data = []
        req_cols = ["planet","distance_from_sun","orbital_eccentricity"]
        for _,data in df.iterrows():
            f = [data.get(col, None) for col in req_cols]
            f[1] = np.float64(f[1]/149.6)
            f[2] = np.float64(f[2])
            final_data.append(f)
        
        return final_data
    
    @staticmethod
    def solar_system2(system_path):
        # read the solar system file
        df = pd.read_csv(system_path)

        final_data = []
        req_cols = ["planet","mass","density"]
        for _,data in df.iterrows():
            f = [data.get(col, None) for col in req_cols]
            f[1] = np.float64(f[1]/5.97)
            f[2] = np.float64(f[2])
            final_data.append(f)
        
        return final_data
    
    @staticmethod
    def solar_system3(system_path):
        # read the solar system file
        df = pd.read_csv(system_path)

        final_data = []
        req_cols = ["planet","distance_from_sun"]
        for _,data in df.iterrows():
            f = [data.get(col, None) for col in req_cols]
            f[1] = np.float64(f[1]/149.6)
            final_data.append(f)
        
        return final_data

class part1:
    @staticmethod
    def bc1(exoplanet_data):
        # doing boxcox transformation and viewing output
        n = len(exoplanet_data)
        arr = np.zeros(n)
        for i in range(n):
            arr[i] = exoplanet_data[i][2]

        transformed, parameter = boxcox(arr)
        return transformed,arr,parameter
    
    @staticmethod
    def bc2(exoplanet_data):
        # doing boxcox transformation and viewing output
        n = len(exoplanet_data)
        arr = np.zeros(n)
        for i in range(n):
            arr[i] = exoplanet_data[i][1]

        transformed, parameter = boxcox(arr)
        return transformed,arr,parameter
    
    @staticmethod
    def mass1(exoplanet_data):
        n = len(exoplanet_data)
        arr = np.zeros(n)
        for i in range(n):
            arr[i] = np.log10(exoplanet_data[i][1])

        return arr
    
    def mass2(exoplanet_data):
        n = len(exoplanet_data)

 
class output:
    @staticmethod
    def output1(transformed_array,original_array):
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.hist(original_array, bins=15, color='skyblue', edgecolor='black')
        plt.title("Original Eccentricities")
        plt.xlabel("Eccentricity")
        plt.ylabel("Frequency")

        plt.subplot(1, 2, 2)
        plt.hist(transformed_array, bins=15, color='lightgreen', edgecolor='black')
        plt.title("Box–Cox Transformed Eccentricities")
        plt.xlabel("Transformed Value")
        plt.ylabel("Frequency")

        plt.show()

    @staticmethod
    def output2(transformed_array):
        plt.figure(figsize=(5, 5))

        plt.hist(transformed_array, bins=15, color='blue', edgecolor='black')
        plt.title("Box–Cox Transformed OrbitalDistance")
        plt.xlabel("Transformed Value")
        plt.ylabel("Frequency")

        plt.show()

    @staticmethod
    def output3(log_array):
        
        plt.figure(figsize=(10, 5))

        # Histogram as PDF
        plt.hist(log_array, bins=30, color='lightblue', edgecolor='black', density=True, alpha=1)

        # Smooth KDE curve
        kde = gaussian_kde(log_array)
        x_vals = np.linspace(min(log_array), max(log_array), 200)
        plt.plot(x_vals, kde(x_vals), color='red', linewidth=2, label='KDE')

        plt.title("Mass distribution of exoplanets (PDF)")
        plt.xlabel("log(m)")
        plt.ylabel("Probability Density")
        plt.legend()
        plt.show()

    @staticmethod
    def output4(planet_data,solar_system):
        names = [p[0] for p in planet_data]
        masses = [p[1] for p in planet_data]
        densities = [p[2] for p in planet_data]

        s_masses = [p[1] for p in solar_system]
        s_densities = [p[2] for p in solar_system]
        # Create scatter plot
        plt.figure(figsize=(10, 5))
        plt.scatter(masses, densities, color='blue', s=5)

        plt.scatter(s_masses, s_densities, color='red', s=5,marker = 's')
        # Set log scales for axes
        plt.xscale("log")
        plt.yscale("log")

        # Labels and title
        plt.xlabel("Mass (Earth Masses, log scale)")
        plt.ylabel("Density (kg/m³, log scale)")
        plt.title("Mass vs Density of Planets (Log–Log Axes)")

        plt.grid(True, which="both", ls="--", alpha=0.6)
        plt.tight_layout()
        plt.show()


cleaned1 = data_extraction.clean_exoplanet_data1("exoplanet_data.csv", "exoplanets_cleaned.csv")
cleaned2 = data_extraction.clean_exoplanet_data2("exoplanet_data.csv","exoplanets_cleaned2.csv")
cleaned3 = data_extraction.clean_exoplanet_data3("exoplanet_data.csv","exoplanets_cleaned3.csv")
solar1 = data_extraction.solar_system1("planets.csv")
solar2 = data_extraction.solar_system2("planets.csv")
solar3 = data_extraction.solar_system3("planets.csv")

print(len(cleaned2))
for data in solar1:
    cleaned1.append(data)

for data in solar2:
    cleaned2.append(data)

for data in solar3:
    cleaned3.append(data)

transformed1,initial1,param1 = part1.bc1(cleaned1)  # boxcox of eccentricities
output.output1(transformed1,initial1)

transformed2,initial2,param2 = part1.bc2(cleaned3)  # boxcox of orbital distance
output.output2(transformed2)

transformed3 = part1.mass1(cleaned2)
output.output3(transformed3)

output.output4(cleaned2,solar2)
# units of cleaned1 (name, distance from star in AU , eccentricity)
# units of cleaned2 (name, mass relative to earth, density in standard units)
# units of cleaned3 (name, distance from star in AU)
'''for data in cleaned1:
    print(data)'''
'''for data in solar1:
     print(data)

for data in solar2:
     print(data)'''

