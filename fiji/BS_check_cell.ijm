// BS_check_cell.ijm  -  check or correct BS-Detector's birth scar for one cell, frame by frame.
//
// For each frame you click the birth-scar midpoint and, if you want to correct them too, the new-pole
// and old-pole tips. Skip a point by pressing OK with nothing selected. Coordinates are saved in raw
// image pixels, one CSV row per frame, in the same layout as the lab's tracking macro.
//
// Use: open the movie in FIJI exactly as it was analysed (all frames, same position, not cropped,
// rotated or resampled). Move the time slider to the first frame to fix (review.csv and `bsd show`
// give the Fiji slice), then Plugins > Macros > Run... and choose this file. Give the cell its
// BS-Detector name (e.g. A01) and save the CSV in the run's corrections/ folder; the next
// `bsd growth` applies it.
//
// The CSV is written after every frame, so stopping early loses nothing.

requires("1.52a");
if (nImages == 0) exit("Open the movie first, then run this macro.");

movie = getTitle();
movieCsv = replace(movie, ",", ";");
movieDir = getInfo("image.directory");
Stack.getDimensions(width, height, channels, slices, frames);
useFrames = frames > 1;
if (useFrames) nT = frames; else nT = slices;
if (nT < 2) exit("This image has only one frame. Open the full time-lapse movie.");
Stack.getPosition(ch, sl, fr);
if (useFrames) startT = fr; else startT = sl;

Dialog.create("Check one cell");
Dialog.addMessage("Movie: " + movie + "  (" + nT + " frames)");
Dialog.addString("BS-Detector cell name (e.g. A01)", "");
Dialog.addString("Your initials (optional)", "");
Dialog.addNumber("First frame to check (as shown on the slider)", startT);
Dialog.addNumber("Then every Nth frame", 1);
Dialog.show();
cell = Dialog.getString();
if (cell == "") exit("Give the BS-Detector cell name, e.g. A01 (see review.csv).");
scorer = Dialog.getString();
startT = Dialog.getNumber();
step = Dialog.getNumber();
if (step < 1) step = 1;

outDir = getDirectory("Choose the run's corrections folder");
base = movie;
dot = lastIndexOf(base, ".");
if (dot > 0) base = substring(base, 0, dot);
base = replace(base, "[^A-Za-z0-9._-]", "_");
outPath = outDir + "fiji_" + base + "_cell-" + cell + ".csv";
if (File.exists(outPath)) {
    if (!getBoolean("A CSV for this cell already exists:\n" + outPath + "\n \nAppend new rows to it?"))
        exit("Stopped. Choose a different cell label to start a new file.");
} else {
    File.saveString("movie,cell_id,scorer,frame_slider,frame_index,scar_x,scar_y,new_pole_x,new_pole_y,old_pole_x,old_pole_y,status,notes\n", outPath);
}

labels = newArray("BIRTH SCAR midpoint (on the centerline, between the two ridges)",
                  "NEW pole tip (optional: only to correct the poles too)",
                  "OLD pole tip (optional: only to correct the poles too)");
colors = newArray("red", "cyan", "yellow");
setTool("point");
t = startT;
nDone = 0;
while (t <= nT) {
    xs = newArray(3); ys = newArray(3); have = newArray(3);
    marksBefore = Overlay.size;
    for (k = 0; k < 3; k++) {
        goTo(t, useFrames);
        run("Select None");
        setTool("point");
        waitForUser("Cell " + cell + "  -  frame " + t + " of " + nT,
            "Click the " + labels[k] + ",\nthen press OK.\n \n" +
            "To skip this point (for example, the scar is not visible yet),\n" +
            "press OK with nothing selected.");
        goTo(t, useFrames);
        if (selectionType() == 10) {
            getSelectionCoordinates(px, py);
            xs[k] = px[px.length - 1]; ys[k] = py[py.length - 1]; have[k] = 1;
            makePoint(xs[k], ys[k]);
            Overlay.addSelection(colors[k]);
            if (useFrames) Overlay.setPosition(0, 0, t); else Overlay.setPosition(t);
        } else {
            have[k] = 0;
        }
        run("Select None");
    }

    Dialog.create("Cell " + cell + "  -  frame " + t);
    Dialog.addChoice("Next step", newArray("Save and go to the next frame",
                                           "Save and finish",
                                           "Redo this frame"), "Save and go to the next frame");
    Dialog.addString("Notes for this frame (optional)", "", 40);
    Dialog.show();
    choice = Dialog.getChoice();
    notes = replace(Dialog.getString(), ",", ";");

    if (choice == "Redo this frame") {
        // Remove the markers added for this frame, then ask again.
        for (i = Overlay.size - 1; i >= marksBefore; i--) Overlay.removeSelection(i);
    } else {
        status = "ok";
        if (choice == "Save and finish") status = "done";
        row = movieCsv + "," + replace(cell, ",", ";") + "," + replace(scorer, ",", ";") + "," + t + "," + (t - 1);
        for (k = 0; k < 3; k++) {
            if (have[k] == 1) row = row + "," + d2s(xs[k], 2) + "," + d2s(ys[k], 2);
            else row = row + ",,";
        }
        row = row + "," + status + "," + notes;
        File.append(row, outPath);
        nDone++;
        if (status != "ok") t = nT + 1;
        else if (t + step > nT) t = nT + 1;
        else t = t + step;
    }
}
showMessage("Done", "Saved " + nDone + " frame(s) for cell " + cell + " to\n" + outPath);

function goTo(t, useFrames) {
    if (useFrames) Stack.setFrame(t); else setSlice(t);
}

