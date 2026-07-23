export const DEFAULT_FONT_SIZE = 12;
export const DEFAULT_FONT_FAMILY = 'sans-serif';
export const DEFAULT_FONT = `${DEFAULT_FONT_SIZE}px ${DEFAULT_FONT_FAMILY}`;

interface Platform {
    // TODO CanvasLike?
    createCanvas(): HTMLCanvasElement
    measureText(text: string, font?: string): { width: number }
    loadImage(
        src: string,
        onload: () => void | HTMLImageElement['onload'],
        onerror: () => void | HTMLImageElement['onerror']
    ): HTMLImageElement
}

// Text width map used for environment there is no canvas
// Only common ascii is used for size concern.

// Generated from following code
//
// ctx.font = '12px sans-serif';
// const asciiRange = [32, 126];
// let mapStr = '';
// for (let i = asciiRange[0]; i <= asciiRange[1]; i++) {
//     const char = String.fromCharCode(i);
//     const width = ctx.measureText(char).width;
//     const ratio = Math.round(width / 12 * 100);
//     mapStr += String.fromCharCode(ratio + 20))
// }
// mapStr.replace(/\\/g, '\\\\');
const OFFSET = 20;
const SCALE = 100;
// TODO other basic fonts?
// eslint-disable-next-line
const defaultWidthMapStr = `007LLmW'55;N0500LLLLLLLLLL00NNNLzWW\\\\WQb\\0FWLg\\bWb\\WQ\\WrWWQ000CL5LLFLL0LL**F*gLLLL5F0LF\\FFF5.5N`;

function getTextWidthMap(mapStr: string): Record<string, number> {
    const map: Record<string, number> = {};
    if (typeof JSON === 'undefined') {
        return map;
    }
    for (let i = 0; i < mapStr.length; i++) {
        const char = String.fromCharCode(i + 32);
        const size = (mapStr.charCodeAt(i) - OFFSET) / SCALE;
        map[char] = size;
    }
    return map;
}

export const DEFAULT_TEXT_WIDTH_MAP = getTextWidthMap(defaultWidthMapStr);

export const platformApi: Platform = {
    // Export methods
    createCanvas() {
        return typeof document !== 'undefined'
            && document.createElement('canvas');
    },

    measureText: (function () {

        let _ctx: CanvasRenderingContext2D;
        let _cachedFont: string;
        return (text: string, font?: string) => {
            if (!_ctx) {
                const canvas = platformApi.createCanvas();
                _ctx = canvas && canvas.getContext('2d');
            }
            if (_ctx) {
                if (_cachedFont !== font) {
                    _cachedFont = _ctx.font = font || DEFAULT_FONT;
                }
                return _ctx.measureText(text);
            }
            else {
                text = text || '';
                font = font || DEFAULT_FONT;
                // Use font size if there is no other method can be used.
                const res = /((?:\d+)?\.?\d*)px/.exec(font);
                const fontSize = res && +res[1] || DEFAULT_FONT_SIZE;
                let width = 0;
                if (font.indexOf('mono') >= 0) {   // is monospace
                    width = fontSize * text.length;
                }
                else {
                    for (let i = 0; i < text.length; i++) {
                        const preCalcWidth = DEFAULT_TEXT_WIDTH_MAP[text[i]];
                        width += preCalcWidth == null ? fontSize : (preCalcWidth * fontSize);
                    }
                }
                return { width };
            }
        };
    })(),

    loadImage(src, onload, onerror) {
        const image = new Image();
        image.onload = onload;
        image.onerror = onerror;
        image.src = src;
        return image;
    }
};

export function setPlatformAPI(newPlatformApis: Partial<Platform>) {
    for (let key in platformApi) {
        // Don't assign unknown methods.
        if ((newPlatformApis as any)[key]) {
            (platformApi as any)[key] = (newPlatformApis as any)[key];
        }
    }
}
