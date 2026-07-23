import LRU from '../core/LRU';
import { extend, isGradientObject, isString, map } from '../core/util';
import { GradientObject } from '../graphic/Gradient';

const kCSSColorTable = {
    'transparent': [0, 0, 0, 0], 'aliceblue': [240, 248, 255, 1],
    'antiquewhite': [250, 235, 215, 1], 'aqua': [0, 255, 255, 1],
    'aquamarine': [127, 255, 212, 1], 'azure': [240, 255, 255, 1],
    'beige': [245, 245, 220, 1], 'bisque': [255, 228, 196, 1],
    'black': [0, 0, 0, 1], 'blanchedalmond': [255, 235, 205, 1],
    'blue': [0, 0, 255, 1], 'blueviolet': [138, 43, 226, 1],
    'brown': [165, 42, 42, 1], 'burlywood': [222, 184, 135, 1],
    'cadetblue': [95, 158, 160, 1], 'chartreuse': [127, 255, 0, 1],
    'chocolate': [210, 105, 30, 1], 'coral': [255, 127, 80, 1],
    'cornflowerblue': [100, 149, 237, 1], 'cornsilk': [255, 248, 220, 1],
    'crimson': [220, 20, 60, 1], 'cyan': [0, 255, 255, 1],
    'darkblue': [0, 0, 139, 1], 'darkcyan': [0, 139, 139, 1],
    'darkgoldenrod': [184, 134, 11, 1], 'darkgray': [169, 169, 169, 1],
    'darkgreen': [0, 100, 0, 1], 'darkgrey': [169, 169, 169, 1],
    'darkkhaki': [189, 183, 107, 1], 'darkmagenta': [139, 0, 139, 1],
    'darkolivegreen': [85, 107, 47, 1], 'darkorange': [255, 140, 0, 1],
    'darkorchid': [153, 50, 204, 1], 'darkred': [139, 0, 0, 1],
    'darksalmon': [233, 150, 122, 1], 'darkseagreen': [143, 188, 143, 1],
    'darkslateblue': [72, 61, 139, 1], 'darkslategray': [47, 79, 79, 1],
    'darkslategrey': [47, 79, 79, 1], 'darkturquoise': [0, 206, 209, 1],
    'darkviolet': [148, 0, 211, 1], 'deeppink': [255, 20, 147, 1],
    'deepskyblue': [0, 191, 255, 1], 'dimgray': [105, 105, 105, 1],
    'dimgrey': [105, 105, 105, 1], 'dodgerblue': [30, 144, 255, 1],
    'firebrick': [178, 34, 34, 1], 'floralwhite': [255, 250, 240, 1],
    'forestgreen': [34, 139, 34, 1], 'fuchsia': [255, 0, 255, 1],
    'gainsboro': [220, 220, 220, 1], 'ghostwhite': [248, 248, 255, 1],
    'gold': [255, 215, 0, 1], 'goldenrod': [218, 165, 32, 1],
    'gray': [128, 128, 128, 1], 'green': [0, 128, 0, 1],
    'greenyellow': [173, 255, 47, 1], 'grey': [128, 128, 128, 1],
    'honeydew': [240, 255, 240, 1], 'hotpink': [255, 105, 180, 1],
    'indianred': [205, 92, 92, 1], 'indigo': [75, 0, 130, 1],
    'ivory': [255, 255, 240, 1], 'khaki': [240, 230, 140, 1],
    'lavender': [230, 230, 250, 1], 'lavenderblush': [255, 240, 245, 1],
    'lawngreen': [124, 252, 0, 1], 'lemonchiffon': [255, 250, 205, 1],
    'lightblue': [173, 216, 230, 1], 'lightcoral': [240, 128, 128, 1],
    'lightcyan': [224, 255, 255, 1], 'lightgoldenrodyellow': [250, 250, 210, 1],
    'lightgray': [211, 211, 211, 1], 'lightgreen': [144, 238, 144, 1],
    'lightgrey': [211, 211, 211, 1], 'lightpink': [255, 182, 193, 1],
    'lightsalmon': [255, 160, 122, 1], 'lightseagreen': [32, 178, 170, 1],
    'lightskyblue': [135, 206, 250, 1], 'lightslategray': [119, 136, 153, 1],
    'lightslategrey': [119, 136, 153, 1], 'lightsteelblue': [176, 196, 222, 1],
    'lightyellow': [255, 255, 224, 1], 'lime': [0, 255, 0, 1],
    'limegreen': [50, 205, 50, 1], 'linen': [250, 240, 230, 1],
    'magenta': [255, 0, 255, 1], 'maroon': [128, 0, 0, 1],
    'mediumaquamarine': [102, 205, 170, 1], 'mediumblue': [0, 0, 205, 1],
    'mediumorchid': [186, 85, 211, 1], 'mediumpurple': [147, 112, 219, 1],
    'mediumseagreen': [60, 179, 113, 1], 'mediumslateblue': [123, 104, 238, 1],
    'mediumspringgreen': [0, 250, 154, 1], 'mediumturquoise': [72, 209, 204, 1],
    'mediumvioletred': [199, 21, 133, 1], 'midnightblue': [25, 25, 112, 1],
    'mintcream': [245, 255, 250, 1], 'mistyrose': [255, 228, 225, 1],
    'moccasin': [255, 228, 181, 1], 'navajowhite': [255, 222, 173, 1],
    'navy': [0, 0, 128, 1], 'oldlace': [253, 245, 230, 1],
    'olive': [128, 128, 0, 1], 'olivedrab': [107, 142, 35, 1],
    'orange': [255, 165, 0, 1], 'orangered': [255, 69, 0, 1],
    'orchid': [218, 112, 214, 1], 'palegoldenrod': [238, 232, 170, 1],
    'palegreen': [152, 251, 152, 1], 'paleturquoise': [175, 238, 238, 1],
    'palevioletred': [219, 112, 147, 1], 'papayawhip': [255, 239, 213, 1],
    'peachpuff': [255, 218, 185, 1], 'peru': [205, 133, 63, 1],
    'pink': [255, 192, 203, 1], 'plum': [221, 160, 221, 1],
    'powderblue': [176, 224, 230, 1], 'purple': [128, 0, 128, 1],
    'red': [255, 0, 0, 1], 'rosybrown': [188, 143, 143, 1],
    'royalblue': [65, 105, 225, 1], 'saddlebrown': [139, 69, 19, 1],
    'salmon': [250, 128, 114, 1], 'sandybrown': [244, 164, 96, 1],
    'seagreen': [46, 139, 87, 1], 'seashell': [255, 245, 238, 1],
    'sienna': [160, 82, 45, 1], 'silver': [192, 192, 192, 1],
    'skyblue': [135, 206, 235, 1], 'slateblue': [106, 90, 205, 1],
    'slategray': [112, 128, 144, 1], 'slategrey': [112, 128, 144, 1],
    'snow': [255, 250, 250, 1], 'springgreen': [0, 255, 127, 1],
    'steelblue': [70, 130, 180, 1], 'tan': [210, 180, 140, 1],
    'teal': [0, 128, 128, 1], 'thistle': [216, 191, 216, 1],
    'tomato': [255, 99, 71, 1], 'turquoise': [64, 224, 208, 1],
    'violet': [238, 130, 238, 1], 'wheat': [245, 222, 179, 1],
    'white': [255, 255, 255, 1], 'whitesmoke': [245, 245, 245, 1],
    'yellow': [255, 255, 0, 1], 'yellowgreen': [154, 205, 50, 1]
};

function clampCssByte(i: number): number {  // Clamp to integer 0 .. 255.
    i = Math.round(i);  // Seems to be what Chrome does (vs truncation).
    return i < 0 ? 0 : i > 255 ? 255 : i;
}

function clampCssAngle(i: number): number {  // Clamp to integer 0 .. 360.
    i = Math.round(i);  // Seems to be what Chrome does (vs truncation).
    return i < 0 ? 0 : i > 360 ? 360 : i;
}

function clampCssFloat(f: number): number {  // Clamp to float 0.0 .. 1.0.
    return f < 0 ? 0 : f > 1 ? 1 : f;
}

function parseCssInt(val: string | number): number {  // int or percentage.
    let str = val as string;
    if (str.length && str.charAt(str.length - 1) === '%') {
        return clampCssByte(parseFloat(str) / 100 * 255);
    }
    return clampCssByte(parseInt(str, 10));
}

function parseCssFloat(val: string | number): number {  // float or percentage.
    let str = val as string;
    if (str.length && str.charAt(str.length - 1) === '%') {
        return clampCssFloat(parseFloat(str) / 100);
    }
    return clampCssFloat(parseFloat(str));
}

function cssHueToRgb(m1: number, m2: number, h: number): number {
    if (h < 0) {
        h += 1;
    }
    else if (h > 1) {
        h -= 1;
    }

    if (h * 6 < 1) {
        return m1 + (m2 - m1) * h * 6;
    }
    if (h * 2 < 1) {
        return m2;
    }
    if (h * 3 < 2) {
        return m1 + (m2 - m1) * (2 / 3 - h) * 6;
    }
    return m1;
}

function lerpNumber(a: number, b: number, p: number): number {
    return a + (b - a) * p;
}

function setRgba(out: number[], r: number, g: number, b: number, a: number): number[] {
    out[0] = r;
    out[1] = g;
    out[2] = b;
    out[3] = a;
    return out;
}
function copyRgba(out: number[], a: number[]) {
    out[0] = a[0];
    out[1] = a[1];
    out[2] = a[2];
    out[3] = a[3];
    return out;
}

const colorCache = new LRU<number[]>(20);
let lastRemovedArr: number[] = null;

function putToCache(colorStr: string, rgbaArr: number[]) {
    // Reuse removed array
    if (lastRemovedArr) {
        copyRgba(lastRemovedArr, rgbaArr);
    }
    lastRemovedArr = colorCache.put(colorStr, lastRemovedArr || (rgbaArr.slice()));
}

export function parse(colorStr: string, rgbaArr?: number[]): number[] {
    if (!colorStr) {
        return;
    }
    rgbaArr = rgbaArr || [];

    let cached = colorCache.get(colorStr);
    if (cached) {
        return copyRgba(rgbaArr, cached);
    }

    // colorStr may be not string
    colorStr = colorStr + '';
    // Remove all whitespace, not compliant, but should just be more accepting.
    let str = colorStr.replace(/ /g, '').toLowerCase();

    // Color keywords (and transparent) lookup.
    if (str in kCSSColorTable) {
        copyRgba(rgbaArr, kCSSColorTable[str as keyof typeof kCSSColorTable]);
        putToCache(colorStr, rgbaArr);
        return rgbaArr;
    }

    // supports the forms #rgb, #rrggbb, #rgba, #rrggbbaa
    // #rrggbbaa(use the last pair of digits as alpha)
    // see https://drafts.csswg.org/css-color/#hex-notation
    const strLen = str.length;
    if (str.charAt(0) === '#') {
        if (strLen === 4 || strLen === 5) {
            const iv = parseInt(str.slice(1, 4), 16);  // TODO(deanm): Stricter parsing.
            if (!(iv >= 0 && iv <= 0xfff)) {
                setRgba(rgbaArr, 0, 0, 0, 1);
                return;  // Covers NaN.
            }
            // interpret values of the form #rgb as #rrggbb and #rgba as #rrggbbaa
            setRgba(rgbaArr,
                ((iv & 0xf00) >> 4) | ((iv & 0xf00) >> 8),
                (iv & 0xf0) | ((iv & 0xf0) >> 4),
                (iv & 0xf) | ((iv & 0xf) << 4),
                strLen === 5 ? parseInt(str.slice(4), 16) / 0xf : 1
            );
            putToCache(colorStr, rgbaArr);
            return rgbaArr;
        }
        else if (strLen === 7 || strLen === 9) {
            const iv = parseInt(str.slice(1, 7), 16);  // TODO(deanm): Stricter parsing.
            if (!(iv >= 0 && iv <= 0xffffff)) {
                setRgba(rgbaArr, 0, 0, 0, 1);
                return;  // Covers NaN.
            }
            setRgba(rgbaArr,
                (iv & 0xff0000) >> 16,
                (iv & 0xff00) >> 8,
                iv & 0xff,
                strLen === 9 ? parseInt(str.slice(7), 16) / 0xff : 1
            );
            putToCache(colorStr, rgbaArr);
            return rgbaArr;
        }

        return;
    }
    let op = str.indexOf('(');
    let ep = str.indexOf(')');
    if (op !== -1 && ep + 1 === strLen) {
        let fname = str.substr(0, op);
        let params: (number | string)[] = str.substr(op + 1, ep - (op + 1)).split(',');
        let alpha = 1;  // To allow case fallthrough.
        switch (fname) {
            case 'rgba':
                if (params.length !== 4) {
                    return params.length === 3
                        // to be compatible with rgb
                        ? setRgba(rgbaArr, +params[0], +params[1], +params[2], 1)
                        : setRgba(rgbaArr, 0, 0, 0, 1);
                }
                alpha = parseCssFloat(params.pop()); // jshint ignore:line
            // Fall through.
            case 'rgb':
                if (params.length >= 3) {
                    setRgba(rgbaArr,
                        parseCssInt(params[0]),
                        parseCssInt(params[1]),
                        parseCssInt(params[2]),
                        params.length === 3 ? alpha : parseCssFloat(params[3])
                    );
                    putToCache(colorStr, rgbaArr);
                    return rgbaArr;
                }
                else {
                    setRgba(rgbaArr, 0, 0, 0, 1);
                    return;
                }
            case 'hsla':
                if (params.length !== 4) {
                    setRgba(rgbaArr, 0, 0, 0, 1);
                    return;
                }
                params[3] = parseCssFloat(params[3]);
                hsla2rgba(params, rgbaArr);
                putToCache(colorStr, rgbaArr);
                return rgbaArr;
            case 'hsl':
                if (params.length !== 3) {
                    setRgba(rgbaArr, 0, 0, 0, 1);
                    return;
                }
                hsla2rgba(params, rgbaArr);
                putToCache(colorStr, rgbaArr);
                return rgbaArr;
            default:
                return;
        }
    }

    setRgba(rgbaArr, 0, 0, 0, 1);
    return;
}

function hsla2rgba(hsla: (number | string) [], rgba?: number[]): number[] {
    const h = (((parseFloat(hsla[0] as string) % 360) + 360) % 360) / 360;  // 0 .. 1
    // NOTE(deanm): According to the CSS spec s/l should only be
    // percentages, but we don't bother and let float or percentage.
    const s = parseCssFloat(hsla[1]);
    const l = parseCssFloat(hsla[2]);
    const m2 = l <= 0.5 ? l * (s + 1) : l + s - l * s;
    const m1 = l * 2 - m2;

    rgba = rgba || [];
    setRgba(rgba,
        clampCssByte(cssHueToRgb(m1, m2, h + 1 / 3) * 255),
        clampCssByte(cssHueToRgb(m1, m2, h) * 255),
        clampCssByte(cssHueToRgb(m1, m2, h - 1 / 3) * 255),
        1
    );

    if (hsla.length === 4) {
        rgba[3] = hsla[3] as number;
    }

    return rgba;
}

function rgba2hsla(rgba: number[]): number[] {
    if (!rgba) {
        return;
    }

    // RGB from 0 to 255
    const R = rgba[0] / 255;
    const G = rgba[1] / 255;
    const B = rgba[2] / 255;

    const vMin = Math.min(R, G, B); // Min. value of RGB
    const vMax = Math.max(R, G, B); // Max. value of RGB
    const delta = vMax - vMin; // Delta RGB value

    const L = (vMax + vMin) / 2;
    let H;
    let S;
    // HSL results from 0 to 1
    if (delta === 0) {
        H = 0;
        S = 0;
    }
    else {
        if (L < 0.5) {
            S = delta / (vMax + vMin);
        }
        else {
            S = delta / (2 - vMax - vMin);
        }

        const deltaR = (((vMax - R) / 6) + (delta / 2)) / delta;
        const deltaG = (((vMax - G) / 6) + (delta / 2)) / delta;
        const deltaB = (((vMax - B) / 6) + (delta / 2)) / delta;

        if (R === vMax) {
            H = deltaB - deltaG;
        }
        else if (G === vMax) {
            H = (1 / 3) + deltaR - deltaB;
        }
        else if (B === vMax) {
            H = (2 / 3) + deltaG - deltaR;
        }

        if (H < 0) {
            H += 1;
        }

        if (H > 1) {
            H -= 1;
        }
    }

    const hsla = [H * 360, S, L];

    if (rgba[3] != null) {
        hsla.push(rgba[3]);
    }

    return hsla;
}

export function lift(color: string, level: number) {
    const colorArr = parse(color);
    if (colorArr) {
        for (let i = 0; i < 3; i++) {
            if (level < 0) {
                colorArr[i] = colorArr[i] * (1 - level) | 0;
            }
            else {
                colorArr[i] = ((255 - colorArr[i]) * level + colorArr[i]) | 0;
            }
            if (colorArr[i] > 255) {
                colorArr[i] = 255;
            }
            else if (colorArr[i] < 0) {
                colorArr[i] = 0;
            }
        }
        return stringify(colorArr, colorArr.length === 4 ? 'rgba' : 'rgb');
    }
}

export function toHex(color: string): string {
    const colorArr = parse(color);
    if (colorArr) {
        return ((1 << 24) + (colorArr[0] << 16) + (colorArr[1] << 8) + (+colorArr[2])).toString(16).slice(1);
    }
}

/**
 * Map value to color. Faster than lerp methods because color is represented by rgba array.
 * @param normalizedValue A float between 0 and 1.
 * @param colors List of rgba color array
 * @param out Mapped gba color array
 * @return will be null/undefined if input illegal.
 */
export function fastLerp(
    normalizedValue: number,
    colors: number[][],
    out?: number[]
): number[] {
    if (!(colors && colors.length)
        || !(normalizedValue >= 0 && normalizedValue <= 1)
    ) {
        return;
    }

    out = out || [];

    const value = normalizedValue * (colors.length - 1);
    const leftIndex = Math.floor(value);
    const rightIndex = Math.ceil(value);
    const leftColor = colors[leftIndex];
    const rightColor = colors[rightIndex];
    const dv = value - leftIndex;
    out[0] = clampCssByte(lerpNumber(leftColor[0], rightColor[0], dv));
    out[1] = clampCssByte(lerpNumber(leftColor[1], rightColor[1], dv));
    out[2] = clampCssByte(lerpNumber(leftColor[2], rightColor[2], dv));
    out[3] = clampCssFloat(lerpNumber(leftColor[3], rightColor[3], dv));

    return out;
}

/**
 * @deprecated
 */
export const fastMapToColor = fastLerp;

type LerpFullOutput = {
    color: string
    leftIndex: number
    rightIndex: number
    value: number
}
/**
 * @param normalizedValue A float between 0 and 1.
 * @param colors Color list.
 * @param fullOutput Default false.
 * @return Result color. If fullOutput,
            return {color: ..., leftIndex: ..., rightIndex: ..., value: ...},
 */
export function lerp(
    normalizedValue: number,
    colors: string[],
    fullOutput: boolean
): LerpFullOutput
export function lerp(
    normalizedValue: number,
    colors: string[]
): string
export function lerp(
    normalizedValue: number,
    colors: string[],
    fullOutput?: boolean
): string | LerpFullOutput {
    if (!(colors && colors.length)
        || !(normalizedValue >= 0 && normalizedValue <= 1)
    ) {
        return;
    }

    const value = normalizedValue * (colors.length - 1);
    const leftIndex = Math.floor(value);
    const rightIndex = Math.ceil(value);
    const leftColor = parse(colors[leftIndex]);
    const rightColor = parse(colors[rightIndex]);
    const dv = value - leftIndex;

    const color = stringify(
        [
            clampCssByte(lerpNumber(leftColor[0], rightColor[0], dv)),
            clampCssByte(lerpNumber(leftColor[1], rightColor[1], dv)),
            clampCssByte(lerpNumber(leftColor[2], rightColor[2], dv)),
            clampCssFloat(lerpNumber(leftColor[3], rightColor[3], dv))
        ],
        'rgba'
    );

    return fullOutput
        ? {
            color: color,
            leftIndex: leftIndex,
            rightIndex: rightIndex,
            value: value
        }
        : color;
}

/**
 * @deprecated
 */
export const mapToColor = lerp;

/**
 * @param color
 * @param h 0 ~ 360, ignore when null.
 * @param s 0 ~ 1, ignore when null.
 * @param l 0 ~ 1, ignore when null.
 * @return Color string in rgba format.
 * @memberOf module:zrender/util/color
 */
export function modifyHSL(color: string, h?: number, s?: number, l?: number): string {
    let colorArr = parse(color);

    if (color) {
        colorArr = rgba2hsla(colorArr);
        h != null && (colorArr[0] = clampCssAngle(h));
        s != null && (colorArr[1] = parseCssFloat(s));
        l != null && (colorArr[2] = parseCssFloat(l));

        return stringify(hsla2rgba(colorArr), 'rgba');
    }
}

/**
 * @param color
 * @param alpha 0 ~ 1
 * @return Color string in rgba format.
 * @memberOf module:zrender/util/color
 */
export function modifyAlpha(color: string, alpha?: number): string {
    const colorArr = parse(color);

    if (colorArr && alpha != null) {
        colorArr[3] = clampCssFloat(alpha);
        return stringify(colorArr, 'rgba');
    }
}

/**
 * @param arrColor like [12,33,44,0.4]
 * @param type 'rgba', 'hsva', ...
 * @return Result color. (If input illegal, return undefined).
 */
export function stringify(arrColor: number[], type: string): string {
    if (!arrColor || !arrColor.length) {
        return;
    }
    let colorStr = arrColor[0] + ',' + arrColor[1] + ',' + arrColor[2];
    if (type === 'rgba' || type === 'hsva' || type === 'hsla') {
        colorStr += ',' + arrColor[3];
    }
    return type + '(' + colorStr + ')';
}

/**
 * Calculate luminance. It will include alpha.
 */
export function lum(color: string, backgroundLum: number) {
    const arr = parse(color);
    return arr
        ? (0.299 * arr[0] + 0.587 * arr[1] + 0.114 * arr[2]) * arr[3] / 255
            + (1 - arr[3]) * backgroundLum  // Blending with assumed white background.
        : 0;
}

/**
 * Generate a random color
 */
export function random(): string {
    return stringify([
        Math.round(Math.random() * 255),
        Math.round(Math.random() * 255),
        Math.round(Math.random() * 255)
    ], 'rgb');
}

const liftedColorCache = new LRU<string>(100);
export function liftColor(color: GradientObject): GradientObject;
export function liftColor(color: string): string;
export function liftColor(color: string | GradientObject): string | GradientObject {
    if (isString(color)) {
        let liftedColor = liftedColorCache.get(color);
        if (!liftedColor) {
            liftedColor = lift(color, -0.1);
            liftedColorCache.put(color, liftedColor);
        }
        return liftedColor;
    }
    else if (isGradientObject(color)) {
        const ret = extend({}, color) as GradientObject;
        ret.colorStops = map(color.colorStops, stop => ({
            offset: stop.offset,
            color: lift(stop.color, -0.1)
        }));
        return ret;
    }
    // Change nothing.
    return color;
}
