from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Example usage
image1_path = "latest_chart.png"  # Replace with your image path
image2_path = "latest_chart2.png"
output_filename = "report.pdf"

class_names = ['can', 'carton', 'plastic bag', 'plastic bottle', 'plastic con', 'styrofoam', 'tire']
class_counts=[1, 3, 5, 4, 6, 1, 0]
danger_levels=['Low(1)', 'Medium(2)', 'High(3)']
danger_counts=[3, 8, 9]
dangers=[ 2.0000,  2.0008,  2.9890,  1.8996,  2.0000,  1.1081,  2.0767,  2.6087 ,  2.9898 ,  3.0000 ,  2.9996 ,  2.0012 ,  2.0019 ,  1.0125 ,  1.6524 ,  2.7370 ,  3.0000 ,  2.9581 ,  2.9937 ,  2.0000 ]

def create_report(image1_path, image2_path, output_filename, class_names, class_counts, danger_levels, danger_counts, dangers):
    # Create a canvas
    c = canvas.Canvas(output_filename, pagesize=letter)
    width, height = letter

    c.setFont("Times-Bold", 20)
    c.drawString(50, height - 50, "Report on Marine Debris")

    c.setFont("Times-Bold", 16)
    c.drawString(50, height - 80, "Marine Debris Classes")

    c.drawImage(image1_path, 50, height - 400, width=400, height=300)

    c.setFont("Times-Bold", 14)
    c.drawString(50, height - 450, "Numbers for each class")

    c.setFont("Times-Roman", 12)
    for i in range(len(class_names)):
        c.drawString(50, height - 470 - 20 * i, f"{class_names[i]}: {class_counts[i]}")

    
    
    c.drawString(50, height - 630, f"Total: {sum(class_counts)}")


    c.showPage()  # Start a new page


    c.setFont("Times-Bold", 16)
    c.drawString(50, height - 80, "Marine Debris Danger Levels")

    c.drawImage(image2_path, 50, height - 400, width=400, height=300)

    c.setFont("Times-Bold", 14)
    c.drawString(50, height - 450, "Numbers for each danger level")

    c.setFont("Times-Roman", 12)
    for i in range(len(danger_levels)):
        c.drawString(50, height - 470 - 20 * i, f"Danger Level {danger_levels[i]}: {danger_counts[i]}")

    c.drawString(50, height - 550, f"Average danger level: {(sum(dangers) / len(dangers)):.2f}")
    c.drawString(50, height - 570, f"Median Danger Level: {(sorted(dangers)[len(dangers) // 2]):.2f}")


    # Save the PDF
    c.save()


create_report(image1_path, image2_path, output_filename, class_names, class_counts, danger_levels, danger_counts, dangers)