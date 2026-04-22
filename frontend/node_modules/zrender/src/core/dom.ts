
import env from './env';
import {buildTransformer} from './fourPointsTransform';
import {Dictionary} from './types';

const EVENT_SAVED_PROP = '___zrEVENTSAVED';
const _calcOut: number[] = [];

type SavedInfo = {
    markers?: HTMLDivElement[]
    trans?: ReturnType<typeof buildTransformer>
    invTrans?: ReturnType<typeof buildTransformer>
    srcCoords?: number[]
}

/**
 * Transform "local coord" from `elFrom` to `elTarget`.
 * "local coord": the coord based on the input `el`. The origin point is at
 *     the position of "left: 0; top: 0;" in the `el`.
 *
 * Support when CSS transform is used.
 *
 * Having the `out` (that is, `[outX, outY]`), we can create an DOM element
 * and set the CSS style as "left: outX; top: outY;" and append it to `elTarge`
 * to locate the element.
 *
 * For example, this code below positions a child of `document.body` on the event
 * point, no matter whether `body` has `margin`/`paddin`/`transfrom`/... :
 * ```js
 * transformLocalCoord(out, container, document.body, event.offsetX, event.offsetY);
 * if (!eqNaN(out[0])) {
 *     // Then locate the tip element on the event point.
 *     var tipEl = document.createElement('div');
 *     tipEl.style.cssText = 'position: absolute; left:' + out[0] + ';top:' + out[1] + ';';
 *     document.body.appendChild(tipEl);
 * }
 * ```
 *
 * Notice: In some env this method is not supported. If called, `out` will be `[NaN, NaN]`.
 *
 * @param {Array.<number>} out [inX: number, inY: number] The output..
 *        If can not transform, `out` will not be modified but return `false`.
 * @param {HTMLElement} elFrom The `[inX, inY]` is based on elFrom.
 * @param {HTMLElement} elTarget The `out` is based on elTarget.
 * @param {number} inX
 * @param {number} inY
 * @return {boolean} Whether transform successfully.
 */
export function transformLocalCoord(
    out: number[],
    elFrom: HTMLElement,
    elTarget: HTMLElement,
    inX: number,
    inY: number
) {
    return transformCoordWithViewport(_calcOut, elFrom, inX, inY, true)
        && transformCoordWithViewport(out, elTarget, _calcOut[0], _calcOut[1]);
}

/**
 * Transform between a "viewport coord" and a "local coord".
 * "viewport coord": the coord based on the left-top corner of the viewport
 *     of the browser.
 * "local coord": the coord based on the input `el`. The origin point is at
 *     the position of "left: 0; top: 0;" in the `el`.
 *
 * Support the case when CSS transform is used on el.
 *
 * @param out [inX: number, inY: number] The output. If `inverse: false`,
 *        it represents "local coord", otherwise "vireport coord".
 *        If can not transform, `out` will not be modified but return `false`.
 * @param el The "local coord" is based on the `el`, see comment above.
 * @param inX If `inverse: false`,
 *        it represents "vireport coord", otherwise "local coord".
 * @param inY If `inverse: false`,
 *        it represents "vireport coord", otherwise "local coord".
 * @param inverse
 *        `true`: from "viewport coord" to "local coord".
 *        `false`: from "local coord" to "viewport coord".
 * @return {boolean} Whether transform successfully.
 */
export function transformCoordWithViewport(
    out: number[],
    el: HTMLElement,
    inX: number,
    inY: number,
    inverse?: boolean
) {
    if (el.getBoundingClientRect && env.domSupported && !isCanvasEl(el)) {
        const saved = (el as any)[EVENT_SAVED_PROP] || ((el as any)[EVENT_SAVED_PROP] = {});
        const markers = prepareCoordMarkers(el, saved);
        const transformer = preparePointerTransformer(markers, saved, inverse);
        if (transformer) {
            transformer(out, inX, inY);
            return true;
        }
    }
    return false;
}

function prepareCoordMarkers(el: HTMLElement, saved: SavedInfo) {
    let markers = saved.markers;
    if (markers) {
        return markers;
    }

    markers = saved.markers = [];
    const propLR = ['left', 'right'];
    const propTB = ['top', 'bottom'];

    for (let i = 0; i < 4; i++) {
        const marker = document.createElement('div');
        const stl = marker.style;
        const idxLR = i % 2;
        const idxTB = (i >> 1) % 2;
        stl.cssText = [
            'position: absolute',
            'visibility: hidden',
            'padding: 0',
            'margin: 0',
            'border-width: 0',
            'user-select: none',
            'width:0',
            'height:0',
            // 'width: 5px',
            // 'height: 5px',
            propLR[idxLR] + ':0',
            propTB[idxTB] + ':0',
            propLR[1 - idxLR] + ':auto',
            propTB[1 - idxTB] + ':auto',
            ''
        ].join('!important;');
        el.appendChild(marker);
        markers.push(marker);
    }

    return markers;
}

function preparePointerTransformer(markers: HTMLDivElement[], saved: SavedInfo, inverse?: boolean) {
    const transformerName: 'invTrans' | 'trans' = inverse ? 'invTrans' : 'trans';
    const transformer = saved[transformerName];
    const oldSrcCoords = saved.srcCoords;
    const srcCoords = [];
    const destCoords = [];
    let oldCoordTheSame = true;

    for (let i = 0; i < 4; i++) {
        const rect = markers[i].getBoundingClientRect();
        const ii = 2 * i;
        const x = rect.left;
        const y = rect.top;
        srcCoords.push(x, y);
        oldCoordTheSame = oldCoordTheSame && oldSrcCoords && x === oldSrcCoords[ii] && y === oldSrcCoords[ii + 1];
        destCoords.push(markers[i].offsetLeft, markers[i].offsetTop);
    }
    // Cache to avoid time consuming of `buildTransformer`.
    return (oldCoordTheSame && transformer)
        ? transformer
        : (
            saved.srcCoords = srcCoords,
            saved[transformerName] = inverse
                ? buildTransformer(destCoords, srcCoords)
                : buildTransformer(srcCoords, destCoords)
        );
}

export function isCanvasEl(el: HTMLElement): el is HTMLCanvasElement {
    return el.nodeName.toUpperCase() === 'CANVAS';
}

const replaceReg = /([&<>"'])/g;
const replaceMap: Dictionary<string> = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    '\'': '&#39;'
};

export function encodeHTML(source: string): string {
    return source == null
        ? ''
        : (source + '').replace(replaceReg, function (str, c) {
            return replaceMap[c];
        });
}
