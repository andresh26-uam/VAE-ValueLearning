package de.hsh.inform.swa.bat4cep.bat;

import java.util.Locale;

import de.hsh.inform.swa.evaluation.RuleWithFitness;
/**
 * Entity class representing a bat with all its features.
 * @author Software Architecture Research Group
 *
 */
public class Bat implements Comparable<Bat> {

    private double f, r, A, v;
    
    private RuleWithFitness solution; // Note: position of a bat == CEP rule. 

    public Bat() {};

    public Bat(double f, double r, double A, double v, RuleWithFitness sol) {
        this.f = f;
        this.r = r;
        this.A = A;
        this.v = v;
        this.solution = sol;
    }

    public double getF() {
        return f;
    }

    public void setF(double f) {
        this.f = f;
    }

    public double getR() {
        return r;
    }

    public void setR(double r) {
        this.r = r;
    }

    public double getA() {
        return A;
    }

    public void setA(double a) {
        A = a;
    }

    public double getV() {
        return v;
    }

    public void setV(double v) {
        this.v = v;
    }

    public RuleWithFitness getSolution() {
        return solution;
    }

    public void setSolution(RuleWithFitness solution) {
        this.solution = solution;
    }

    public Bat copy() {
        return new Bat(getF(), getR(), getA(), getV(), getSolution().copy());
    }

    @Override
    public int compareTo(Bat o) {
        return Double.compare(solution.getTotalFitness(), o.solution.getTotalFitness());
    }

    @Override
    public boolean equals(Object o) {
        Bat r = (Bat) o;
        if (this.solution.toString().equals(r.solution.toString())) return true;
        return false;
    }

    @Override
    public String toString() {
        return String.format(Locale.ENGLISH, "f: %5f r: %5f A: %5f v: %5f \n%s", f, r, A, v, solution.toString());
    }
}
