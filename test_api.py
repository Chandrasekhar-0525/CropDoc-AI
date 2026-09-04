import requests

image_path = input("Enter image path: ")
crop = input("Enter crop (cotton/soybean/sugarcane/sorghum): ").lower()

with open(image_path, "rb") as image:

    response = requests.post(
        "http://127.0.0.1:5000/predict",
        data={"crop": crop},
        files={"image": image}
    )

print("\n==============================")
print("        AI API RESULT")
print("==============================")

print(response.json())

print("==============================")