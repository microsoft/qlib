import { MatrixArray } from './matrix';

export interface PointLike {
    x: number
    y: number
}
export default class Point {

    x: number

    y: number

    constructor(x?: number, y?: number) {
        this.x = x || 0;
        this.y = y || 0;
    }

    /**
     * Copy from another point
     */
    copy(other: PointLike) {
        this.x = other.x;
        this.y = other.y;
        return this;
    }

    /**
     * Clone a point
     */
    clone() {
        return new Point(this.x, this.y);
    }

    /**
     * Set x and y
     */
    set(x: number, y: number) {
        this.x = x;
        this.y = y;
        return this;
    }

    /**
     * If equal to another point
     */
    equal(other: PointLike) {
        return other.x === this.x && other.y === this.y;
    }

    /**
     * Add another point
     */
    add(other: PointLike) {
        this.x += other.x;
        this.y += other.y;
        return this;
    }

    scale(scalar: number) {
        this.x *= scalar;
        this.y *= scalar;
    }

    scaleAndAdd(other: PointLike, scalar: number) {
        this.x += other.x * scalar;
        this.y += other.y * scalar;
    }

    /**
     * Sub another point
     */
    sub(other: PointLike) {
        this.x -= other.x;
        this.y -= other.y;
        return this;
    }

    /**
     * Dot product with other point
     */
    dot(other: PointLike) {
        return this.x * other.x + this.y * other.y;
    }

    /**
     * Get length of point
     */
    len() {
        return Math.sqrt(this.x * this.x + this.y * this.y);
    }

    /**
     * Get squared length
     */
    lenSquare() {
        return this.x * this.x + this.y * this.y;
    }

    /**
     * Normalize
     */
    normalize() {
        const len = this.len();
        this.x /= len;
        this.y /= len;
        return this;
    }

    /**
     * Distance to another point
     */
    distance(other: PointLike) {
        const dx = this.x - other.x;
        const dy = this.y - other.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    /**
     * Square distance to another point
     */
    distanceSquare(other: Point) {
        const dx = this.x - other.x;
        const dy = this.y - other.y;
        return dx * dx + dy * dy;
    }

    /**
     * Negate
     */
    negate() {
        this.x = -this.x;
        this.y = -this.y;
        return this;
    }

    /**
     * Apply a transform matrix array.
     */
    transform(m: MatrixArray) {
        if (!m) {
            return;
        }
        const x = this.x;
        const y = this.y;
        this.x = m[0] * x + m[2] * y + m[4];
        this.y = m[1] * x + m[3] * y + m[5];
        return this;
    }

    toArray(out: number[]) {
        out[0] = this.x;
        out[1] = this.y;
        return out;
    }

    fromArray(input: number[]) {
        this.x = input[0];
        this.y = input[1];
    }

    static set(p: PointLike, x: number, y: number) {
        p.x = x;
        p.y = y;
    }

    static copy(p: PointLike, p2: PointLike) {
        p.x = p2.x;
        p.y = p2.y;
    }

    static len(p: PointLike) {
        return Math.sqrt(p.x * p.x + p.y * p.y);
    }

    static lenSquare(p: PointLike) {
        return p.x * p.x + p.y * p.y;
    }

    static dot(p0: PointLike, p1: PointLike) {
        return p0.x * p1.x + p0.y * p1.y;
    }

    static add(out: PointLike, p0: PointLike, p1: PointLike) {
        out.x = p0.x + p1.x;
        out.y = p0.y + p1.y;
    }

    static sub(out: PointLike, p0: PointLike, p1: PointLike) {
        out.x = p0.x - p1.x;
        out.y = p0.y - p1.y;
    }

    static scale(out: PointLike, p0: PointLike, scalar: number) {
        out.x = p0.x * scalar;
        out.y = p0.y * scalar;
    }

    static scaleAndAdd(out: PointLike, p0: PointLike, p1: PointLike, scalar: number) {
        out.x = p0.x + p1.x * scalar;
        out.y = p0.y + p1.y * scalar;
    }

    static lerp(out: PointLike, p0: PointLike, p1: PointLike, t: number) {
        const onet = 1 - t;
        out.x = onet * p0.x + t * p1.x;
        out.y = onet * p0.y + t * p1.y;
    }
}