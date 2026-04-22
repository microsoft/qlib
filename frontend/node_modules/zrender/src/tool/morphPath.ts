import PathProxy from '../core/PathProxy';
import { cubicSubdivide } from '../core/curve';
import Path from '../graphic/Path';
import Element, { ElementAnimateConfig } from '../Element';
import { defaults, map } from '../core/util';
import { lerp } from '../core/vector';
import Group, { GroupLike } from '../graphic/Group';
import { clonePath } from './path';
import { MatrixArray } from '../core/matrix';
import Transformable from '../core/Transformable';
import { ZRenderType } from '../zrender';
import { split } from './dividePath';
import { pathToBezierCurves } from './convertPath';

function alignSubpath(subpath1: number[], subpath2: number[]): [number[], number[]] {
    const len1 = subpath1.length;
    const len2 = subpath2.length;
    if (len1 === len2) {
        return [subpath1, subpath2];
    }
    const tmpSegX: number[] = [];
    const tmpSegY: number[] = [];

    const shorterPath = len1 < len2 ? subpath1 : subpath2;
    const shorterLen = Math.min(len1, len2);
    // Should divide excatly
    const diff = Math.abs(len2 - len1) / 6;
    const shorterBezierCount = (shorterLen - 2) / 6;
    // Add `diff` number of beziers
    const eachCurveSubDivCount = Math.ceil(diff / shorterBezierCount) + 1;

    const newSubpath = [shorterPath[0], shorterPath[1]];
    let remained = diff;

    for (let i = 2; i < shorterLen;) {
        let x0 = shorterPath[i - 2];
        let y0 = shorterPath[i - 1];
        let x1 = shorterPath[i++];
        let y1 = shorterPath[i++];
        let x2 = shorterPath[i++];
        let y2 = shorterPath[i++];
        let x3 = shorterPath[i++];
        let y3 = shorterPath[i++];

        if (remained <= 0) {
            newSubpath.push(x1, y1, x2, y2, x3, y3);
            continue;
        }

        let actualSubDivCount = Math.min(remained, eachCurveSubDivCount - 1) + 1;
        for (let k = 1; k <= actualSubDivCount; k++) {
            const p = k / actualSubDivCount;

            cubicSubdivide(x0, x1, x2, x3, p, tmpSegX);
            cubicSubdivide(y0, y1, y2, y3, p, tmpSegY);

            // tmpSegX[3] === tmpSegX[4]
            x0 = tmpSegX[3];
            y0 = tmpSegY[3];

            newSubpath.push(tmpSegX[1], tmpSegY[1], tmpSegX[2], tmpSegY[2], x0, y0);
            x1 = tmpSegX[5];
            y1 = tmpSegY[5];
            x2 = tmpSegX[6];
            y2 = tmpSegY[6];
            // The last point (x3, y3) is still the same.
        }
        remained -= actualSubDivCount - 1;
    }

    return shorterPath === subpath1 ? [newSubpath, subpath2] : [subpath1, newSubpath];
}

function createSubpath(lastSubpathSubpath: number[], otherSubpath: number[]) {
    const len = lastSubpathSubpath.length;
    const lastX = lastSubpathSubpath[len - 2];
    const lastY = lastSubpathSubpath[len - 1];

    const newSubpath: number[] = [];
    for (let i = 0; i < otherSubpath.length;) {
        newSubpath[i++] = lastX;
        newSubpath[i++] = lastY;
    }
    return newSubpath;
}

/**
 * Make two bezier arrays aligns on structure. To have better animation.
 *
 * It will:
 * Make two bezier arrays have same number of subpaths.
 * Make each subpath has equal number of bezier curves.
 *
 * array is the convert result of pathToBezierCurves.
 */
export function alignBezierCurves(array1: number[][], array2: number[][]) {

    let lastSubpath1;
    let lastSubpath2;

    let newArray1 = [];
    let newArray2 = [];

    for (let i = 0; i < Math.max(array1.length, array2.length); i++) {
        const subpath1 = array1[i];
        const subpath2 = array2[i];

        let newSubpath1;
        let newSubpath2;

        if (!subpath1) {
            newSubpath1 = createSubpath(lastSubpath1 || subpath2, subpath2);
            newSubpath2 = subpath2;
        }
        else if (!subpath2) {
            newSubpath2 = createSubpath(lastSubpath2 || subpath1, subpath1);
            newSubpath1 = subpath1;
        }
        else {
            [newSubpath1, newSubpath2] = alignSubpath(subpath1, subpath2);
            lastSubpath1 = newSubpath1;
            lastSubpath2 = newSubpath2;
        }

        newArray1.push(newSubpath1);
        newArray2.push(newSubpath2);
    }

    return [newArray1, newArray2];
}

interface MorphingPath extends Path {
    __morphT: number;
}

export interface CombineMorphingPath extends Path {
    childrenRef(): (CombineMorphingPath | Path)[]
    __isCombineMorphing: boolean;
}

export function centroid(array: number[]) {
    // https://en.wikipedia.org/wiki/Centroid#Of_a_polygon
    let signedArea = 0;
    let cx = 0;
    let cy = 0;
    const len = array.length;
    // Polygon should been closed.
    for (let i = 0, j = len - 2; i < len; j = i, i += 2) {
        const x0 = array[j];
        const y0 = array[j + 1];
        const x1 = array[i];
        const y1 = array[i + 1];
        const a = x0 * y1 - x1 * y0;
        signedArea += a;
        cx += (x0 + x1) * a;
        cy += (y0 + y1) * a;
    }

    if (signedArea === 0) {
        return [array[0] || 0, array[1] || 0];
    }

    return [cx / signedArea / 3, cy / signedArea / 3, signedArea];
}

/**
 * Offset the points to find the nearest morphing distance.
 * Return beziers count needs to be offset.
 */
function findBestRingOffset(
    fromSubBeziers: number[],
    toSubBeziers: number[],
    fromCp: number[],
    toCp: number[]
) {
    const bezierCount = (fromSubBeziers.length - 2) / 6;
    let bestScore = Infinity;
    let bestOffset = 0;

    const len = fromSubBeziers.length;
    const len2 = len - 2;
    for (let offset = 0; offset < bezierCount; offset++) {
        const cursorOffset = offset * 6;
        let score = 0;

        for (let k = 0; k < len; k += 2) {
            let idx = k === 0 ? cursorOffset : ((cursorOffset + k - 2) % len2 + 2);

            const x0 = fromSubBeziers[idx] - fromCp[0];
            const y0 = fromSubBeziers[idx + 1] - fromCp[1];
            const x1 = toSubBeziers[k] - toCp[0];
            const y1 = toSubBeziers[k + 1] - toCp[1];

            const dx = x1 - x0;
            const dy = y1 - y0;
            score += dx * dx + dy * dy;
        }
        if (score < bestScore) {
            bestScore = score;
            bestOffset = offset;
        }
    }

    return bestOffset;
}

function reverse(array: number[]) {
    const newArr: number[] = [];
    const len = array.length;
    for (let i = 0; i < len; i += 2) {
        newArr[i] = array[len - i - 2];
        newArr[i + 1] = array[len - i - 1];
    }
    return newArr;
}

type MorphingData = {
    from: number[];
    to: number[];
    fromCp: number[];
    toCp: number[];
    rotation: number;
}[];

/**
 * If we interpolating between two bezier curve arrays.
 * It will have many broken effects during the transition.
 * So we try to apply an extra rotation which can make each bezier curve morph as small as possible.
 */
function findBestMorphingRotation(
    fromArr: number[][],
    toArr: number[][],
    searchAngleIteration: number,
    searchAngleRange: number
): MorphingData {
    const result = [];

    let fromNeedsReverse: boolean;

    for (let i = 0; i < fromArr.length; i++) {
        let fromSubpathBezier = fromArr[i];
        const toSubpathBezier = toArr[i];

        const fromCp = centroid(fromSubpathBezier);
        const toCp = centroid(toSubpathBezier);

        if (fromNeedsReverse == null) {
            // Reverse from array if two have different directions.
            // Determine the clockwise based on the first subpath.
            // Reverse all subpaths or not. Avoid winding rule changed.
            fromNeedsReverse = fromCp[2] < 0 !== toCp[2] < 0;
        }

        const newFromSubpathBezier: number[] = [];
        const newToSubpathBezier: number[] = [];
        let bestAngle = 0;
        let bestScore = Infinity;
        let tmpArr: number[] = [];

        const len = fromSubpathBezier.length;
        if (fromNeedsReverse) {
            // Make sure clockwise
            fromSubpathBezier = reverse(fromSubpathBezier);
        }
        const offset = findBestRingOffset(fromSubpathBezier, toSubpathBezier, fromCp, toCp) * 6;

        const len2 = len - 2;
        for (let k = 0; k < len2; k += 2) {
            // Not include the start point.
            const idx = (offset + k) % len2 + 2;
            newFromSubpathBezier[k + 2] = fromSubpathBezier[idx] - fromCp[0];
            newFromSubpathBezier[k + 3] = fromSubpathBezier[idx + 1] - fromCp[1];
        }
        newFromSubpathBezier[0] = fromSubpathBezier[offset] - fromCp[0];
        newFromSubpathBezier[1] = fromSubpathBezier[offset + 1] - fromCp[1];

        if (searchAngleIteration > 0) {
            const step = searchAngleRange / searchAngleIteration;
            for (let angle = -searchAngleRange / 2; angle <= searchAngleRange / 2; angle += step) {
                const sa = Math.sin(angle);
                const ca = Math.cos(angle);
                let score = 0;

                for (let k = 0; k < fromSubpathBezier.length; k += 2) {
                    const x0 = newFromSubpathBezier[k];
                    const y0 = newFromSubpathBezier[k + 1];
                    const x1 = toSubpathBezier[k] - toCp[0];
                    const y1 = toSubpathBezier[k + 1] - toCp[1];

                    // Apply rotation on the target point.
                    const newX1 = x1 * ca - y1 * sa;
                    const newY1 = x1 * sa + y1 * ca;

                    tmpArr[k] = newX1;
                    tmpArr[k + 1] = newY1;

                    const dx = newX1 - x0;
                    const dy = newY1 - y0;

                    // Use dot product to have min direction change.
                    // const d = Math.sqrt(x0 * x0 + y0 * y0);
                    // score += x0 * dx / d + y0 * dy / d;
                    score += dx * dx + dy * dy;
                }

                if (score < bestScore) {
                    bestScore = score;
                    bestAngle = angle;
                    // Copy.
                    for (let m = 0; m < tmpArr.length; m++) {
                        newToSubpathBezier[m] = tmpArr[m];
                    }
                }
            }
        }
        else {
            for (let i = 0; i < len; i += 2) {
                newToSubpathBezier[i] = toSubpathBezier[i] - toCp[0];
                newToSubpathBezier[i + 1] = toSubpathBezier[i + 1] - toCp[1];
            }
        }

        result.push({
            from: newFromSubpathBezier,
            to: newToSubpathBezier,
            fromCp,
            toCp,
            rotation: -bestAngle
        });
    }
    return result;
}

export function isCombineMorphing(path: Element): path is CombineMorphingPath {
    return (path as CombineMorphingPath).__isCombineMorphing;
}

export function isMorphing(el: Element) {
    return (el as MorphingPath).__morphT >= 0;
}

const SAVED_METHOD_PREFIX = '__mOriginal_';
function saveAndModifyMethod<T extends object, M extends keyof T>(
    obj: T,
    methodName: M,
    modifiers: { replace?: T[M], after?: T[M], before?: T[M] }
) {
    const savedMethodName = SAVED_METHOD_PREFIX + methodName;
    const originalMethod = (obj as any)[savedMethodName] || obj[methodName];
    if (!(obj as any)[savedMethodName]) {
        (obj as any)[savedMethodName] = obj[methodName];
    }
    const replace = modifiers.replace;
    const after = modifiers.after;
    const before = modifiers.before;

    (obj as any)[methodName] = function () {
        const args = arguments;
        let res;
        before && (before as unknown as Function).apply(this, args);
        // Still call the original method if not replacement.
        if (replace) {
            res = (replace as unknown as Function).apply(this, args);
        }
        else {
            res = originalMethod.apply(this, args);
        }
        after && (after as unknown as Function).apply(this, args);
        return res;
    };
}
function restoreMethod<T extends object>(
    obj: T,
    methodName: keyof T
) {
    const savedMethodName = SAVED_METHOD_PREFIX + methodName;
    if ((obj as any)[savedMethodName]) {
        obj[methodName] = (obj as any)[savedMethodName];
        (obj as any)[savedMethodName] = null;
    }
}

function applyTransformOnBeziers(bezierCurves: number[][], mm: MatrixArray) {
    for (let i = 0; i < bezierCurves.length; i++) {
        const subBeziers = bezierCurves[i];
        for (let k = 0; k < subBeziers.length;) {
            const x = subBeziers[k];
            const y = subBeziers[k + 1];

            subBeziers[k++] = mm[0] * x + mm[2] * y + mm[4];
            subBeziers[k++] = mm[1] * x + mm[3] * y + mm[5];
        }
    }
}

function prepareMorphPath(
    fromPath: Path,
    toPath: Path
) {
    const fromPathProxy = fromPath.getUpdatedPathProxy();
    const toPathProxy = toPath.getUpdatedPathProxy();

    const [fromBezierCurves, toBezierCurves] =
        alignBezierCurves(pathToBezierCurves(fromPathProxy), pathToBezierCurves(toPathProxy));

    const fromPathTransform = fromPath.getComputedTransform();
    const toPathTransform = toPath.getComputedTransform();
    function updateIdentityTransform(this: Transformable) {
        this.transform = null;
    }
    fromPathTransform && applyTransformOnBeziers(fromBezierCurves, fromPathTransform);
    toPathTransform && applyTransformOnBeziers(toBezierCurves, toPathTransform);
    // Just ignore transform
    saveAndModifyMethod(toPath, 'updateTransform', { replace: updateIdentityTransform });
    toPath.transform = null;

    const morphingData = findBestMorphingRotation(fromBezierCurves, toBezierCurves, 10, Math.PI);

    const tmpArr: number[] = [];

    saveAndModifyMethod(toPath, 'buildPath', { replace(path: PathProxy) {
        const t = (toPath as MorphingPath).__morphT;
        const onet = 1 - t;

        const newCp: number[] = [];

        for (let i = 0; i < morphingData.length; i++) {
            const item = morphingData[i];
            const from = item.from;
            const to = item.to;
            const angle = item.rotation * t;
            const fromCp = item.fromCp;
            const toCp = item.toCp;
            const sa = Math.sin(angle);
            const ca = Math.cos(angle);

            lerp(newCp, fromCp, toCp, t);

            for (let m = 0; m < from.length; m += 2) {
                const x0 = from[m];
                const y0 = from[m + 1];
                const x1 = to[m];
                const y1 = to[m + 1];

                const x = x0 * onet + x1 * t;
                const y = y0 * onet + y1 * t;

                tmpArr[m] = (x * ca - y * sa) + newCp[0];
                tmpArr[m + 1] = (x * sa + y * ca) + newCp[1];
            }

            let x0 = tmpArr[0];
            let y0 = tmpArr[1];

            path.moveTo(x0, y0);

            for (let m = 2; m < from.length;) {
                const x1 = tmpArr[m++];
                const y1 = tmpArr[m++];
                const x2 = tmpArr[m++];
                const y2 = tmpArr[m++];
                const x3 = tmpArr[m++];
                const y3 = tmpArr[m++];

                // Is a line.
                if (x0 === x1 && y0 === y1 && x2 === x3 && y2 === y3) {
                    path.lineTo(x3, y3);
                }
                else {
                    path.bezierCurveTo(x1, y1, x2, y2, x3, y3);
                }
                x0 = x3;
                y0 = y3;
            }
        }
    } });
}

/**
 * Morphing from old path to new path.
 */
export function morphPath(
    fromPath: Path,
    toPath: Path,
    animationOpts: ElementAnimateConfig
): Path {
    if (!fromPath || !toPath) {
        return toPath;
    }

    const oldDone = animationOpts.done;
    // const oldAborted = animationOpts.aborted;
    const oldDuring = animationOpts.during;

    prepareMorphPath(fromPath, toPath);

    (toPath as MorphingPath).__morphT = 0;

    function restoreToPath() {
        restoreMethod(toPath, 'buildPath');
        restoreMethod(toPath, 'updateTransform');
        // Mark as not in morphing
        (toPath as MorphingPath).__morphT = -1;
        // Cleanup.
        toPath.createPathProxy();
        toPath.dirtyShape();
    }

    toPath.animateTo({
        __morphT: 1
    } as any, defaults({
        during(p) {
            toPath.dirtyShape();
            oldDuring && oldDuring(p);
        },
        done() {
            restoreToPath();
            oldDone && oldDone();
        }
        // NOTE: Don't do restore if aborted.
        // Because all status was just set when animation started.
        // aborted() {
        //     oldAborted && oldAborted();
        // }
    } as ElementAnimateConfig, animationOpts));

    return toPath;
}

// https://github.com/mapbox/earcut/blob/master/src/earcut.js#L437
// https://jsfiddle.net/pissang/2jk7x145/
// function zOrder(x: number, y: number, minX: number, minY: number, maxX: number, maxY: number) {
//     // Normalize coords to 0 - 1
//     // The transformed into non-negative 15-bit integer range
//     x = (maxX === minX) ? 0 : Math.round(32767 * (x - minX) / (maxX - minX));
//     y = (maxY === minY) ? 0 : Math.round(32767 * (y - minY) / (maxY - minY));

//     x = (x | (x << 8)) & 0x00FF00FF;
//     x = (x | (x << 4)) & 0x0F0F0F0F;
//     x = (x | (x << 2)) & 0x33333333;
//     x = (x | (x << 1)) & 0x55555555;

//     y = (y | (y << 8)) & 0x00FF00FF;
//     y = (y | (y << 4)) & 0x0F0F0F0F;
//     y = (y | (y << 2)) & 0x33333333;
//     y = (y | (y << 1)) & 0x55555555;

//     return x | (y << 1);
// }

// https://github.com/w8r/hilbert/blob/master/hilbert.js#L30
// https://jsfiddle.net/pissang/xdnbzg6v/
function hilbert(x: number, y: number, minX: number, minY: number, maxX: number, maxY: number) {
    const bits = 16;
    x = (maxX === minX) ? 0 : Math.round(32767 * (x - minX) / (maxX - minX));
    y = (maxY === minY) ? 0 : Math.round(32767 * (y - minY) / (maxY - minY));

    let d = 0;
    let tmp: number;
    for (let s = (1 << bits) / 2; s > 0; s /= 2) {
        let rx = 0;
        let ry = 0;

        if ((x & s) > 0) {
            rx = 1;
        }
        if ((y & s) > 0) {
            ry = 1;
        }

        d += s * s * ((3 * rx) ^ ry);

        if (ry === 0) {
            if (rx === 1) {
                x = s - 1 - x;
                y = s - 1 - y;
            }
            tmp = x;
            x = y;
            y = tmp;
        }
    }
    return d;
}

// Sort paths on hilbert. Not using z-order because it may still have large cross.
// So the left most source can animate to the left most target, not right most target.
// Hope in this way. We can make sure each element is animated to the proper target. Not the farthest.
function sortPaths(pathList: Path[]): Path[] {
    let xMin = Infinity;
    let yMin = Infinity;
    let xMax = -Infinity;
    let yMax = -Infinity;
    const cps = map(pathList, path => {
        const rect = path.getBoundingRect();
        const m = path.getComputedTransform();
        const x = rect.x + rect.width / 2 + (m ? m[4] : 0);
        const y = rect.y + rect.height / 2 + (m ? m[5] : 0);
        xMin = Math.min(x, xMin);
        yMin = Math.min(y, yMin);
        xMax = Math.max(x, xMax);
        yMax = Math.max(y, yMax);
        return [x, y];
    });

    const items = map(cps, (cp, idx) => {
        return {
            cp,
            z: hilbert(cp[0], cp[1], xMin, yMin, xMax, yMax),
            path: pathList[idx]
        };
    });

    return items.sort((a, b) => a.z - b.z).map(item => item.path);
}

export interface DividePathParams {
    path: Path,
    count: number
};
export interface DividePath {
    (params: DividePathParams): Path[]
}

export interface IndividualDelay {
    (index: number, count: number, fromPath: Path, toPath: Path): number
}

function defaultDividePath(param: DividePathParams) {
    return split(param.path, param.count);
}
export interface CombineConfig extends ElementAnimateConfig {
    /**
     * Transform of returned will be ignored.
     */
    dividePath?: DividePath
    /**
     * delay of each individual.
     * Because individual are sorted on z-order. The index is also sorted top-left / right-down.
     */
    individualDelay?: IndividualDelay
    /**
     * If sort splitted paths so the movement between them can be more natural
     */
    // sort?: boolean
}

function createEmptyReturn() {
    return {
        fromIndividuals: [] as Path[],
        toIndividuals: [] as Path[],
        count: 0
    };
}
/**
 * Make combine morphing from many paths to one.
 * Will return a group to replace the original path.
 */
export function combineMorph(
    fromList: (CombineMorphingPath | Path)[],
    toPath: Path,
    animationOpts: CombineConfig
) {
    let fromPathList: Path[] = [];

    function addFromPath(fromList: Element[]) {
        for (let i = 0; i < fromList.length; i++) {
            const from = fromList[i];
            if (isCombineMorphing(from)) {
                addFromPath((from as GroupLike).childrenRef());
            }
            else if (from instanceof Path) {
                fromPathList.push(from);
            }
        }
    }
    addFromPath(fromList);

    const separateCount = fromPathList.length;

    // fromPathList.length is 0.
    if (!separateCount) {
        return createEmptyReturn();
    }

    const dividePath = animationOpts.dividePath || defaultDividePath;

    let toSubPathList = dividePath({
        path: toPath, count: separateCount
    });
    if (toSubPathList.length !== separateCount) {
        console.error('Invalid morphing: unmatched splitted path');
        return createEmptyReturn();
    }

    fromPathList = sortPaths(fromPathList);
    toSubPathList = sortPaths(toSubPathList);

    const oldDone = animationOpts.done;
    // const oldAborted = animationOpts.aborted;
    const oldDuring = animationOpts.during;
    const individualDelay = animationOpts.individualDelay;

    const identityTransform = new Transformable();

    for (let i = 0; i < separateCount; i++) {
        const from = fromPathList[i];
        const to = toSubPathList[i];

        to.parent = toPath as unknown as Group;

        // Ignore transform in each subpath.
        to.copyTransform(identityTransform);

        // Will do morphPath for each individual if individualDelay is set.
        if (!individualDelay) {
            prepareMorphPath(from, to);
        }
    }

    (toPath as CombineMorphingPath).__isCombineMorphing = true;
    (toPath as CombineMorphingPath).childrenRef = function () {
        return toSubPathList;
    };

    function addToSubPathListToZr(zr: ZRenderType) {
        for (let i = 0; i < toSubPathList.length; i++) {
            toSubPathList[i].addSelfToZr(zr);
        }
    }
    saveAndModifyMethod(toPath, 'addSelfToZr', {
        after(zr) {
            addToSubPathListToZr(zr);
        }
    });
    saveAndModifyMethod(toPath, 'removeSelfFromZr', {
        after(zr) {
            for (let i = 0; i < toSubPathList.length; i++) {
                toSubPathList[i].removeSelfFromZr(zr);
            }
        }
    });

    function restoreToPath() {
        (toPath as CombineMorphingPath).__isCombineMorphing = false;
        // Mark as not in morphing
        (toPath as MorphingPath).__morphT = -1;
        (toPath as CombineMorphingPath).childrenRef = null;

        restoreMethod(toPath, 'addSelfToZr');
        restoreMethod(toPath, 'removeSelfFromZr');
    }

    const toLen = toSubPathList.length;

    if (individualDelay) {
        let animating = toLen;
        const eachDone = () => {
            animating--;
            if (animating === 0) {
                restoreToPath();
                oldDone && oldDone();
            }
        };
        // Animate each element individually.
        for (let i = 0; i < toLen; i++) {
            // TODO only call during once?
            const indivdualAnimationOpts = individualDelay ? defaults({
                delay: (animationOpts.delay || 0) + individualDelay(i, toLen, fromPathList[i], toSubPathList[i]),
                done: eachDone
            } as ElementAnimateConfig, animationOpts) : animationOpts;
            morphPath(fromPathList[i], toSubPathList[i], indivdualAnimationOpts);
        }
    }
    else {
        (toPath as MorphingPath).__morphT = 0;
        toPath.animateTo({
            __morphT: 1
        } as any, defaults({
            during(p) {
                for (let i = 0; i < toLen; i++) {
                    const child = toSubPathList[i] as MorphingPath;
                    child.__morphT = (toPath as MorphingPath).__morphT;
                    child.dirtyShape();
                }
                oldDuring && oldDuring(p);
            },
            done() {
                restoreToPath();
                for (let i = 0; i < fromList.length; i++) {
                    restoreMethod(fromList[i], 'updateTransform');
                }
                oldDone && oldDone();
            }
        } as ElementAnimateConfig, animationOpts));
    }

    if (toPath.__zr) {
        addToSubPathListToZr(toPath.__zr);
    }

    return {
        fromIndividuals: fromPathList,
        toIndividuals: toSubPathList,
        count: toLen
    };
}
export interface SeparateConfig extends ElementAnimateConfig {
    dividePath?: DividePath
    individualDelay?: IndividualDelay
    /**
     * If sort splitted paths so the movement between them can be more natural
     */
    // sort?: boolean
    // // If the from path of separate animation is doing combine animation.
    // // And the paths number is not same with toPathList. We need to do enter/leave animation
    // // on the missing/spare paths.
    // enter?: (el: Path) => void
    // leave?: (el: Path) => void
}

/**
 * Make separate morphing from one path to many paths.
 * Make the MorphingKind of `toPath` become `'ONE_ONE'`.
 */
export function separateMorph(
    fromPath: Path,
    toPathList: Path[],
    animationOpts: SeparateConfig
) {
    const toLen = toPathList.length;
    let fromPathList: Path[] = [];

    const dividePath = animationOpts.dividePath || defaultDividePath;

    function addFromPath(fromList: Element[]) {
        for (let i = 0; i < fromList.length; i++) {
            const from = fromList[i];
            if (isCombineMorphing(from)) {
                addFromPath((from as GroupLike).childrenRef());
            }
            else if (from instanceof Path) {
                fromPathList.push(from);
            }
        }
    }
    // This case most happen when a combining path is called to reverse the animation
    // to its original separated state.
    if (isCombineMorphing(fromPath)) {
        addFromPath(fromPath.childrenRef());

        const fromLen = fromPathList.length;
        if (fromLen < toLen) {
            let k = 0;
            for (let i = fromLen; i < toLen; i++) {
                // Create a clone
                fromPathList.push(clonePath(fromPathList[k++ % fromLen]));
            }
        }
        // Else simply remove if fromLen > toLen
        fromPathList.length = toLen;
    }
    else {
        fromPathList = dividePath({ path: fromPath, count: toLen });
        const fromPathTransform = fromPath.getComputedTransform();
        for (let i = 0; i < fromPathList.length; i++) {
            // Force use transform of source path.
            fromPathList[i].setLocalTransform(fromPathTransform);
        }
        if (fromPathList.length !== toLen) {
            console.error('Invalid morphing: unmatched splitted path');
            return createEmptyReturn();
        }
    }

    fromPathList = sortPaths(fromPathList);
    toPathList = sortPaths(toPathList);

    const individualDelay = animationOpts.individualDelay;
    for (let i = 0; i < toLen; i++) {
        const indivdualAnimationOpts = individualDelay ? defaults({
            delay: (animationOpts.delay || 0) + individualDelay(i, toLen, fromPathList[i], toPathList[i])
        } as ElementAnimateConfig, animationOpts) : animationOpts;
        morphPath(fromPathList[i], toPathList[i], indivdualAnimationOpts);
    }

    return {
        fromIndividuals: fromPathList,
        toIndividuals: toPathList,
        count: toPathList.length
    };
}

export { split as defaultDividePath };
