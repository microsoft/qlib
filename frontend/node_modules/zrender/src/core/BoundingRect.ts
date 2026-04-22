/**
 * @module echarts/core/BoundingRect
 */

import * as matrix from './matrix';
import Point, { PointLike } from './Point';

const mathMin = Math.min;
const mathMax = Math.max;

const lt = new Point();
const rb = new Point();
const lb = new Point();
const rt = new Point();

const minTv = new Point();
const maxTv = new Point();

class BoundingRect {

    x: number
    y: number
    width: number
    height: number

    constructor(x: number, y: number, width: number, height: number) {
        if (width < 0) {
            x = x + width;
            width = -width;
        }
        if (height < 0) {
            y = y + height;
            height = -height;
        }

        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
    }

    union(other: BoundingRect) {
        const x = mathMin(other.x, this.x);
        const y = mathMin(other.y, this.y);

        // If x is -Infinity and width is Infinity (like in the case of
        // IncrementalDisplayble), x + width would be NaN
        if (isFinite(this.x) && isFinite(this.width)) {
            this.width = mathMax(
                other.x + other.width,
                this.x + this.width
            ) - x;
        }
        else {
            this.width = other.width;
        }

        if (isFinite(this.y) && isFinite(this.height)) {
            this.height = mathMax(
                other.y + other.height,
                this.y + this.height
            ) - y;
        }
        else {
            this.height = other.height;
        }

        this.x = x;
        this.y = y;
    }

    applyTransform(m: matrix.MatrixArray) {
        BoundingRect.applyTransform(this, this, m);
    }

    calculateTransform(b: RectLike): matrix.MatrixArray {
        const a = this;
        const sx = b.width / a.width;
        const sy = b.height / a.height;

        const m = matrix.create();

        // 矩阵右乘
        matrix.translate(m, m, [-a.x, -a.y]);
        matrix.scale(m, m, [sx, sy]);
        matrix.translate(m, m, [b.x, b.y]);

        return m;
    }

    intersect(b: RectLike, mtv?: PointLike): boolean {
        if (!b) {
            return false;
        }

        if (!(b instanceof BoundingRect)) {
            // Normalize negative width/height.
            b = BoundingRect.create(b);
        }

        const a = this;
        const ax0 = a.x;
        const ax1 = a.x + a.width;
        const ay0 = a.y;
        const ay1 = a.y + a.height;

        const bx0 = b.x;
        const bx1 = b.x + b.width;
        const by0 = b.y;
        const by1 = b.y + b.height;

        let overlap = !(ax1 < bx0 || bx1 < ax0 || ay1 < by0 || by1 < ay0);
        if (mtv) {
            let dMin = Infinity;
            let dMax = 0;
            const d0 = Math.abs(ax1 - bx0);
            const d1 = Math.abs(bx1 - ax0);
            const d2 = Math.abs(ay1 - by0);
            const d3 = Math.abs(by1 - ay0);
            const dx = Math.min(d0, d1);
            const dy = Math.min(d2, d3);
            // On x axis
            if (ax1 < bx0 || bx1 < ax0) {
                if (dx > dMax) {
                    dMax = dx;
                    if (d0 < d1) {
                        Point.set(maxTv, -d0, 0); // b is on the right
                    }
                    else {
                        Point.set(maxTv, d1, 0);  // b is on the left
                    }
                }
            }
            else {
                if (dx < dMin) {
                    dMin = dx;
                    if (d0 < d1) {
                        Point.set(minTv, d0, 0); // b is on the right
                    }
                    else {
                        Point.set(minTv, -d1, 0);  // b is on the left
                    }
                }
            }

            // On y axis
            if (ay1 < by0 || by1 < ay0) {
                if (dy > dMax) {
                    dMax = dy;
                    if (d2 < d3) {
                        Point.set(maxTv, 0, -d2); // b is on the bottom(larger y)
                    }
                    else {
                        Point.set(maxTv, 0, d3);  // b is on the top(smaller y)
                    }
                }
            }
            else {
                if (dx < dMin) {
                    dMin = dx;
                    if (d2 < d3) {
                        Point.set(minTv, 0, d2); // b is on the bottom
                    }
                    else {
                        Point.set(minTv, 0, -d3);  // b is on the top
                    }
                }
            }
        }

        if (mtv) {
            Point.copy(mtv, overlap ? minTv : maxTv);
        }
        return overlap;
    }

    contain(x: number, y: number): boolean {
        const rect = this;
        return x >= rect.x
            && x <= (rect.x + rect.width)
            && y >= rect.y
            && y <= (rect.y + rect.height);
    }

    clone() {
        return new BoundingRect(this.x, this.y, this.width, this.height);
    }

    /**
     * Copy from another rect
     */
    copy(other: RectLike) {
        BoundingRect.copy(this, other);
    }

    plain(): RectLike {
        return {
            x: this.x,
            y: this.y,
            width: this.width,
            height: this.height
        };
    }

    /**
     * If not having NaN or Infinity with attributes
     */
    isFinite(): boolean {
        return isFinite(this.x)
            && isFinite(this.y)
            && isFinite(this.width)
            && isFinite(this.height);
    }

    isZero(): boolean {
        return this.width === 0 || this.height === 0;
    }

    static create(rect: RectLike): BoundingRect {
        return new BoundingRect(rect.x, rect.y, rect.width, rect.height);
    }

    static copy(target: RectLike, source: RectLike) {
        target.x = source.x;
        target.y = source.y;
        target.width = source.width;
        target.height = source.height;
    }

    static applyTransform(target: RectLike, source: RectLike, m: matrix.MatrixArray) {
        // In case usage like this
        // el.getBoundingRect().applyTransform(el.transform)
        // And element has no transform
        if (!m) {
            if (target !== source) {
                BoundingRect.copy(target, source);
            }
            return;
        }
        // Fast path when there is no rotation in matrix.
        if (m[1] < 1e-5 && m[1] > -1e-5 && m[2] < 1e-5 && m[2] > -1e-5) {
            const sx = m[0];
            const sy = m[3];
            const tx = m[4];
            const ty = m[5];
            target.x = source.x * sx + tx;
            target.y = source.y * sy + ty;
            target.width = source.width * sx;
            target.height = source.height * sy;
            if (target.width < 0) {
                target.x += target.width;
                target.width = -target.width;
            }
            if (target.height < 0) {
                target.y += target.height;
                target.height = -target.height;
            }
            return;
        }

        // source and target can be same instance.
        lt.x = lb.x = source.x;
        lt.y = rt.y = source.y;
        rb.x = rt.x = source.x + source.width;
        rb.y = lb.y = source.y + source.height;

        lt.transform(m);
        rt.transform(m);
        rb.transform(m);
        lb.transform(m);

        target.x = mathMin(lt.x, rb.x, lb.x, rt.x);
        target.y = mathMin(lt.y, rb.y, lb.y, rt.y);
        const maxX = mathMax(lt.x, rb.x, lb.x, rt.x);
        const maxY = mathMax(lt.y, rb.y, lb.y, rt.y);
        target.width = maxX - target.x;
        target.height = maxY - target.y;
    }
}


export type RectLike = {
    x: number
    y: number
    width: number
    height: number
}

export default BoundingRect;