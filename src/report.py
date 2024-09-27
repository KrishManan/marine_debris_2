from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Example usage
image1_path = "latest_chart.png"  # Replace with your image path
image2_path = "latest_chart2.png"
output_filename = "report.pdf"

class_names = ['can', 'carton', 'plastic bag', 'plastic bottle', 'plastic con', 'styrofoam', 'tire']
class_counts=[3,4,5,6,7,8,8]
danger_levels=[1,2,3]
danger_counts=[4,5,6]
dangers=[1,2,3,2,1,2,3,4,5,6]

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


    c.showPage()  # Start a new page


    c.setFont("Times-Bold", 16)
    c.drawString(50, height - 80, "Marine Debris Danger Levels")

    c.drawImage(image2_path, 50, height - 400, width=400, height=300)

    c.setFont("Times-Bold", 14)
    c.drawString(50, height - 450, "Numbers for each danger level")

    c.setFont("Times-Roman", 12)
    for i in range(len(danger_levels)):
        c.drawString(50, height - 470 - 20 * i, f"Danger Level {danger_levels[i]}: {danger_counts[i]}")

    c.drawString(50, height - 550, f"Average danger level: {sum(dangers) / len(dangers)}")


    # Save the PDF
    c.save()


