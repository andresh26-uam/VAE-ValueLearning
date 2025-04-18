package de.hsh.inform.swa.bat4cep.bat.update;

import java.util.concurrent.ThreadLocalRandom;

import de.hsh.inform.swa.cep.Rule;
import de.hsh.inform.swa.cep.windows.LengthWindow;
import de.hsh.inform.swa.cep.windows.TimeWindow;
import de.hsh.inform.swa.cep.windows.Window;
import de.hsh.inform.swa.util.builder.WindowBuilder;

/**
 * Class that coordinates all changes to the window of a given rule.
 * @author Software Architecture Research Group
 *
 */
public class WindowUpdate {
    static void update(Rule rule, WindowBuilder wb) {
        Window newWindow = ThreadLocalRandom.current().nextDouble() < 0.5 ? wb.getRandomLengthWindow() : wb.getRandomTimeWindow();
        long value= newWindow.getValue() < rule.getWindow().getMinValue() ?  rule.getWindow().getMinValue(): newWindow.getValue();
        newWindow.setValue(value);
        rule.setWindow(newWindow);
    }

    static void explicitReduceWindowSize(Rule rule) {
        Window win = rule.getWindow().copy();
        long length = win.getValue();
        long newLengthBound = win.getMinValue() + length;
        long value= ThreadLocalRandom.current().nextLong(win.getMinValue(), newLengthBound <= win.getMinValue() ? win.getMinValue()+1 : newLengthBound);
        win.setValue(value);
        rule.setWindow(win);
    }

    static void localRandomWindow(Rule rule, WindowBuilder wb, double radiusFactor) {
        Window win = rule.getWindow().copy();
        long radius = 0;
        long lowerBound = 0;
        long upperBound = 0;
        long newWindow = 0;
        long length = win.getValue();
        if (win instanceof TimeWindow) {
            radius = (long) ((wb.getMaxTimeDifference() - wb.getMinTimeDifference()) * radiusFactor); 
            upperBound = length + radius < wb.getMaxTimeDifference() ? length + radius : wb.getMaxTimeDifference(); 
        }else if (win instanceof LengthWindow) {
            radius = (long) ((wb.getMaxLength() - wb.getMinLength()) * 0.2);
            upperBound = length + radius < wb.getMaxLength() ? length + radius : wb.getMaxLength();
        }
        lowerBound = length - radius > win.getMinValue() ? length - radius : win.getMinValue(); 
        newWindow = ThreadLocalRandom.current().nextLong(lowerBound, upperBound <= lowerBound ? lowerBound + 1 : upperBound);
        
        win.setValue(newWindow);
        rule.setWindow(win);
    }
}
