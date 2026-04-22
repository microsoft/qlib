import * as matrix from './matrix';
import * as vector from './vector';

const mIdentity = matrix.identity;

const EPSILON = 5e-5;

function isNotAroundZero(val: number) {
    return val > EPSILON || val < -EPSILON;
}

const scaleTmp: vector.VectorArray = [];
const tmpTransform: matrix.MatrixArray = [];
const originTransform = matrix.create();
const abs = Math.abs;

class Transformable {

    parent: Transformable

    x: number
    y: number

    scaleX: number
    scaleY: number

    skewX: number
    skewY: number

    rotation: number

    /**
     * Will translated the element to the anchor position before applying other transforms.
     */
    anchorX: number
    anchorY: number
    /**
     * Origin of scale, rotation, skew
     */
    originX: number
    originY: number

    /**
     * Scale ratio
     */
    globalScaleRatio: number

    transform: matrix.MatrixArray
    invTransform: matrix.MatrixArray

    /**
     * Get computed local transform
     */
    getLocalTransform(m?: matrix.MatrixArray) {
        return Transformable.getLocalTransform(this, m);
    }

    /**
     * Set position from array
     */
    setPosition(arr: number[]) {
        this.x = arr[0];
        this.y = arr[1];
    }
    /**
     * Set scale from array
     */
    setScale(arr: number[]) {
        this.scaleX = arr[0];
        this.scaleY = arr[1];
    }

    /**
     * Set skew from array
     */
    setSkew(arr: number[]) {
        this.skewX = arr[0];
        this.skewY = arr[1];
    }

    /**
     * Set origin from array
     */
    setOrigin(arr: number[]) {
        this.originX = arr[0];
        this.originY = arr[1];
    }

    /**
     * If needs to compute transform
     */
    needLocalTransform(): boolean {
        return isNotAroundZero(this.rotation)
            || isNotAroundZero(this.x)
            || isNotAroundZero(this.y)
            || isNotAroundZero(this.scaleX - 1)
            || isNotAroundZero(this.scaleY - 1)
            || isNotAroundZero(this.skewX)
            || isNotAroundZero(this.skewY);
    }

    /**
     * Update global transform
     */
    updateTransform() {
        const parentTransform = this.parent && this.parent.transform;
        const needLocalTransform = this.needLocalTransform();

        let m = this.transform;
        if (!(needLocalTransform || parentTransform)) {
            if (m) {
                mIdentity(m);
                // reset invTransform
                this.invTransform = null;
            }
            return;
        }

        m = m || matrix.create();

        if (needLocalTransform) {
            this.getLocalTransform(m);
        }
        else {
            mIdentity(m);
        }

        // 应用父节点变换
        if (parentTransform) {
            if (needLocalTransform) {
                matrix.mul(m, parentTransform, m);
            }
            else {
                matrix.copy(m, parentTransform);
            }
        }
        // 保存这个变换矩阵
        this.transform = m;

        this._resolveGlobalScaleRatio(m);
    }

    private _resolveGlobalScaleRatio(m: matrix.MatrixArray) {
        const globalScaleRatio = this.globalScaleRatio;
        if (globalScaleRatio != null && globalScaleRatio !== 1) {
            this.getGlobalScale(scaleTmp);
            const relX = scaleTmp[0] < 0 ? -1 : 1;
            const relY = scaleTmp[1] < 0 ? -1 : 1;
            const sx = ((scaleTmp[0] - relX) * globalScaleRatio + relX) / scaleTmp[0] || 0;
            const sy = ((scaleTmp[1] - relY) * globalScaleRatio + relY) / scaleTmp[1] || 0;

            m[0] *= sx;
            m[1] *= sx;
            m[2] *= sy;
            m[3] *= sy;
        }

        this.invTransform = this.invTransform || matrix.create();
        matrix.invert(this.invTransform, m);
    }

    /**
     * Get computed global transform
     * NOTE: this method will force update transform on all ancestors.
     * Please be aware of the potential performance cost.
     */
    getComputedTransform() {
        let transformNode: Transformable = this;
        const ancestors: Transformable[] = [];
        while (transformNode) {
            ancestors.push(transformNode);
            transformNode = transformNode.parent;
        }

        // Update from topdown.
        while (transformNode = ancestors.pop()) {
            transformNode.updateTransform();
        }

        return this.transform;
    }

    setLocalTransform(m: vector.VectorArray) {
        if (!m) {
            // TODO return or set identity?
            return;
        }
        let sx = m[0] * m[0] + m[1] * m[1];
        let sy = m[2] * m[2] + m[3] * m[3];

        const rotation = Math.atan2(m[1], m[0]);

        const shearX = Math.PI / 2 + rotation - Math.atan2(m[3], m[2]);
        sy = Math.sqrt(sy) * Math.cos(shearX);
        sx = Math.sqrt(sx);

        this.skewX = shearX;
        this.skewY = 0;
        this.rotation = -rotation;

        this.x = +m[4];
        this.y = +m[5];
        this.scaleX = sx;
        this.scaleY = sy;

        this.originX = 0;
        this.originY = 0;
    }
    /**
     * 分解`transform`矩阵到`position`, `rotation`, `scale`
     */
    decomposeTransform() {
        if (!this.transform) {
            return;
        }
        const parent = this.parent;
        let m = this.transform;
        if (parent && parent.transform) {
            // Get local transform and decompose them to position, scale, rotation
            parent.invTransform = parent.invTransform || matrix.create();
            matrix.mul(tmpTransform, parent.invTransform, m);
            m = tmpTransform;
        }
        const ox = this.originX;
        const oy = this.originY;
        if (ox || oy) {
            originTransform[4] = ox;
            originTransform[5] = oy;
            matrix.mul(tmpTransform, m, originTransform);
            tmpTransform[4] -= ox;
            tmpTransform[5] -= oy;
            m = tmpTransform;
        }

        this.setLocalTransform(m);
    }

    /**
     * Get global scale
     */
    getGlobalScale(out?: vector.VectorArray): vector.VectorArray {
        const m = this.transform;
        out = out || [];
        if (!m) {
            out[0] = 1;
            out[1] = 1;
            return out;
        }
        out[0] = Math.sqrt(m[0] * m[0] + m[1] * m[1]);
        out[1] = Math.sqrt(m[2] * m[2] + m[3] * m[3]);
        if (m[0] < 0) {
            out[0] = -out[0];
        }
        if (m[3] < 0) {
            out[1] = -out[1];
        }
        return out;
    }
    /**
     * 变换坐标位置到 shape 的局部坐标空间
     */
    transformCoordToLocal(x: number, y: number): number[] {
        const v2 = [x, y];
        const invTransform = this.invTransform;
        if (invTransform) {
            vector.applyTransform(v2, v2, invTransform);
        }
        return v2;
    }

    /**
     * 变换局部坐标位置到全局坐标空间
     */
    transformCoordToGlobal(x: number, y: number): number[] {
        const v2 = [x, y];
        const transform = this.transform;
        if (transform) {
            vector.applyTransform(v2, v2, transform);
        }
        return v2;
    }


    getLineScale() {
        const m = this.transform;
        // Get the line scale.
        // Determinant of `m` means how much the area is enlarged by the
        // transformation. So its square root can be used as a scale factor
        // for width.
        return m && abs(m[0] - 1) > 1e-10 && abs(m[3] - 1) > 1e-10
            ? Math.sqrt(abs(m[0] * m[3] - m[2] * m[1]))
            : 1;
    }

    copyTransform(source: Transformable) {
        copyTransform(this, source);
    }


    static getLocalTransform(target: Transformable, m?: matrix.MatrixArray): matrix.MatrixArray {
        m = m || [];

        const ox = target.originX || 0;
        const oy = target.originY || 0;
        const sx = target.scaleX;
        const sy = target.scaleY;
        const ax = target.anchorX;
        const ay = target.anchorY;
        const rotation = target.rotation || 0;
        const x = target.x;
        const y = target.y;
        const skewX = target.skewX ? Math.tan(target.skewX) : 0;
        // TODO: zrender use different hand in coordinate system and y axis is inversed.
        const skewY = target.skewY ? Math.tan(-target.skewY) : 0;

        // The order of transform (-anchor * -origin * scale * skew * rotate * origin * translate).
        // We merge (-origin * scale * skew) into one. Also did identity in these operations.
        // origin
        if (ox || oy || ax || ay) {
            const dx = ox + ax;
            const dy = oy + ay;
            m[4] = -dx * sx - skewX * dy * sy;
            m[5] = -dy * sy - skewY * dx * sx;
        }
        else {
            m[4] = m[5] = 0;
        }
        // scale
        m[0] = sx;
        m[3] = sy;
        // skew
        m[1] = skewY * sx;
        m[2] = skewX * sy;

        // Apply rotation
        rotation && matrix.rotate(m, m, rotation);

        // Translate back from origin and apply translation
        m[4] += ox + x;
        m[5] += oy + y;

        return m;
    }

    private static initDefaultProps = (function () {
        const proto = Transformable.prototype;
        proto.scaleX =
        proto.scaleY =
        proto.globalScaleRatio = 1;
        proto.x =
        proto.y =
        proto.originX =
        proto.originY =
        proto.skewX =
        proto.skewY =
        proto.rotation =
        proto.anchorX =
        proto.anchorY = 0;
    })()
};

export const TRANSFORMABLE_PROPS = [
    'x', 'y', 'originX', 'originY', 'anchorX', 'anchorY', 'rotation', 'scaleX', 'scaleY', 'skewX', 'skewY'
] as const;

export type TransformProp = (typeof TRANSFORMABLE_PROPS)[number]

export function copyTransform(
    target: Partial<Pick<Transformable, TransformProp>>,
    source: Pick<Transformable, TransformProp>
) {
    for (let i = 0; i < TRANSFORMABLE_PROPS.length; i++) {
        const propName = TRANSFORMABLE_PROPS[i];
        target[propName] = source[propName];
    }
}

export default Transformable;