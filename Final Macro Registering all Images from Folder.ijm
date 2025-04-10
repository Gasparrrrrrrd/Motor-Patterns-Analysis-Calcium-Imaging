  // Ensure an image is selected  
// Select the folder containing images
folder = getDirectory("Select a folder with images");
list = getFileList(folder);

outputfolder = getDirectory("Select output folder for images");

for (i = 3; i < list.length; i++) {
    if (endsWith(list[i], ".tif")) {  // Adjust file extension if needed
        open(folder + list[i]); // Open the image
        imageID = getImageID();  
        imageTitle = getTitle();  

		run("Align slices in stack...", "method=5 windowsizex=229 windowsizey=101 x0=8 y0=71 swindow=0 subpixel=false itpmethod=0 ref.slice=350 show=true");        // Save the processed image
        
        outputPath = outputfolder + File.separator + replace(list[i], ".tif", "_register.tif");
        saveAs("Tiff", outputPath);

        // Close all images
        close("*");
    }
}