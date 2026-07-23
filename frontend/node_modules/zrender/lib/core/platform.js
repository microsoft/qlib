export var DEFAULT_FONT_SIZE = 12;
export var DEFAULT_FONT_FAMILY = 'sans-serif';
export var DEFAULT_FONT = DEFAULT_FONT_SIZE + "px " + DEFAULT_FONT_FAMILY;
var OFFSET = 20;
var SCALE = 100;
var defaultWidthMapStr = "007LLmW'55;N0500LLLLLLLLLL00NNNLzWW\\\\WQb\\0FWLg\\bWb\\WQ\\WrWWQ000CL5LLFLL0LL**F*gLLLL5F0LF\\FFF5.5N";
function getTextWidthMap(mapStr) {
    var map = {};
    if (typeof JSON === 'undefined') {
        return map;
    }
    for (var i = 0; i < mapStr.length; i++) {
        var char = String.fromCharCode(i + 32);
        var size = (mapStr.charCodeAt(i) - OFFSET) / SCALE;
        map[char] = size;
    }
    return map;
}
export var DEFAULT_TEXT_WIDTH_MAP = getTextWidthMap(defaultWidthMapStr);
export var platformApi = {
    createCanvas: function () {
        return typeof document !== 'undefined'
            && document.createElement('canvas');
    },
    measureText: (function () {
        var _ctx;
        var _cachedFont;
        return function (text, font) {
            if (!_ctx) {
                var canvas = platformApi.createCanvas();
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
                var res = /((?:\d+)?\.?\d*)px/.exec(font);
                var fontSize = res && +res[1] || DEFAULT_FONT_SIZE;
                var width = 0;
                if (font.indexOf('mono') >= 0) {
                    width = fontSize * text.length;
                }
                else {
                    for (var i = 0; i < text.length; i++) {
                        var preCalcWidth = DEFAULT_TEXT_WIDTH_MAP[text[i]];
                        width += preCalcWidth == null ? fontSize : (preCalcWidth * fontSize);
                    }
                }
                return { width: width };
            }
        };
    })(),
    loadImage: function (src, onload, onerror) {
        var image = new Image();
        image.onload = onload;
        image.onerror = onerror;
        image.src = src;
        return image;
    }
};
export function setPlatformAPI(newPlatformApis) {
    for (var key in platformApi) {
        if (newPlatformApis[key]) {
            platformApi[key] = newPlatformApis[key];
        }
    }
}
