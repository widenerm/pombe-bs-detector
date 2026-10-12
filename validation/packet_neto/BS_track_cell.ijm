// BS_track_cell.ijm  -  follow ONE cell from birth to division and record,
// in each sampled frame, three points: the birth-scar midpoint, the new-pole
// tip and the old-pole tip.  Coordinates are saved in raw image pixels
// (independent of any spatial calibration), one CSV row per frame.
//
// Use: open the movie in FIJI (all frames, not cropped or rotated), move the
// time slider to the cell's birth frame, then Plugins > Macros > Run... and
// choose this file.  Run it again for each cell.
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

Dialog.create("Track one cell");
Dialog.addMessage("Movie: " + movie + "  (" + nT + " frames)");
Dialog.addString("Cell label (your own, e.g. c1)", "c1");
Dialog.addString("Your initials", "cr");
Dialog.addNumber("Birth frame (as shown on the slider)", startT);
Dialog.addNumber("Annotate every Nth frame", 2);
Dialog.show();
cell = Dialog.getString();
scorer = Dialog.getString();
startT = Dialog.getNumber();
step = Dialog.getNumber();
if (step < 1) step = 1;

outDir = getDirectory("Choose the folder to save this cell's CSV in");
base = movie;
dot = lastIndexOf(base, ".");
if (dot > 0) base = substring(base, 0, dot);
base = replace(base, "[^A-Za-z0-9._-]", "_");
outPath = outDir + base + "_cell-" + cell + "_" + scorer + ".csv";
if (File.exists(outPath)) {
    if (!getBoolean("A CSV for this cell already exists:\n" + outPath + "\n \nAppend new rows to it?"))
        exit("Stopped. Choose a different cell label to start a new file.");
} else {
    File.saveString("movie,cell_id,scorer,frame_slider,frame_index,scar_x,scar_y,new_pole_x,new_pole_y,old_pole_x,old_pole_y,status,notes\n", outPath);
}

labels = newArray("BIRTH SCAR midpoint (on the centerline, between the two ridges)",
                  "NEW pole tip (the end made at the last division)",
                  "OLD pole tip (the other end)");
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
                                           "Save; the cell DIVIDED in this frame (finish)",
                                           "Save; cell lost or ambiguous (finish)",
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
        if (startsWith(choice, "Save; the cell DIVIDED")) status = "divided";
        if (startsWith(choice, "Save; cell lost")) status = "lost";
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

