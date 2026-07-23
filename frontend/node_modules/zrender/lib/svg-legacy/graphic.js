import { createElement, XLINKNS } from '../svg/core.js';
import { getMatrixStr, TEXT_ALIGN_TO_ANCHOR, adjustTextY } from '../svg/helper.js';
import { getLineHeight } from '../contain/text.js';
import SVGPathRebuilder from '../svg/SVGPathRebuilder.js';
import mapStyleToAttrs from '../svg/mapStyleToAttrs.js';
import { DEFAULT_FONT } from '../core/platform.js';
function setTransform(svgEl, m) {
    if (m) {
        attr(svgEl, 'transform', getMatrixStr(m));
    }
}
function attr(el, key, val) {
    if (!val || val.type !== 'linear' && val.type !== 'radial') {
        el.setAttribute(key, val);
    }
}
function attrXLink(el, key, val) {
    el.setAttributeNS(XLINKNS, key, val);
}
function attrXML(el, key, val) {
    el.setAttributeNS('http://www.w3.org/XML/1998/namespace', key, val);
}
function bindStyle(svgEl, style, el) {
    mapStyleToAttrs(function (key, val) { return attr(svgEl, key, val); }, style, el, true);
}
var svgPath = {
    brush: function (el) {
        var style = el.style;
        var svgEl = el.__svgEl;
        if (!svgEl) {
            svgEl = createElement('path');
            el.__svgEl = svgEl;
        }
        if (!el.path) {
            el.createPathProxy();
        }
        var path = el.path;
        if (el.shapeChanged()) {
            path.beginPath();
            el.buildPath(path, el.shape);
            el.pathUpdated();
        }
        var pathVersion = path.getVersion();
        var elExt = el;
        var svgPathBuilder = elExt.__svgPathBuilder;
        if (elExt.__svgPathVersion !== pathVersion || !svgPathBuilder || el.style.strokePercent < 1) {
            if (!svgPathBuilder) {
                svgPathBuilder = elExt.__svgPathBuilder = new SVGPathRebuilder();
            }
            svgPathBuilder.reset();
            path.rebuildPath(svgPathBuilder, el.style.strokePercent);
            svgPathBuilder.generateStr();
            elExt.__svgPathVersion = pathVersion;
        }
        attr(svgEl, 'd', svgPathBuilder.getStr());
        bindStyle(svgEl, style, el);
        setTransform(svgEl, el.transform);
    }
};
export { svgPath as path };
var svgImage = {
    brush: function (el) {
        var style = el.style;
        var image = style.image;
        if (image instanceof HTMLImageElement) {
            image = image.src;
        }
        else if (image instanceof HTMLCanvasElement) {
            image = image.toDataURL();
        }
        if (!image) {
            return;
        }
        var x = style.x || 0;
        var y = style.y || 0;
        var dw = style.width;
        var dh = style.height;
        var svgEl = el.__svgEl;
        if (!svgEl) {
            svgEl = createElement('image');
            el.__svgEl = svgEl;
        }
        if (image !== el.__imageSrc) {
            attrXLink(svgEl, 'href', image);
            el.__imageSrc = image;
        }
        attr(svgEl, 'width', dw + '');
        attr(svgEl, 'height', dh + '');
        attr(svgEl, 'x', x + '');
        attr(svgEl, 'y', y + '');
        bindStyle(svgEl, style, el);
        setTransform(svgEl, el.transform);
    }
};
export { svgImage as image };
var svgText = {
    brush: function (el) {
        var style = el.style;
        var text = style.text;
        text != null && (text += '');
        if (!text || isNaN(style.x) || isNaN(style.y)) {
            return;
        }
        var textSvgEl = el.__svgEl;
        if (!textSvgEl) {
            textSvgEl = createElement('text');
            attrXML(textSvgEl, 'xml:space', 'preserve');
            el.__svgEl = textSvgEl;
        }
        var font = style.font || DEFAULT_FONT;
        var textSvgElStyle = textSvgEl.style;
        textSvgElStyle.font = font;
        textSvgEl.textContent = text;
        bindStyle(textSvgEl, style, el);
        setTransform(textSvgEl, el.transform);
        var x = style.x || 0;
        var y = adjustTextY(style.y || 0, getLineHeight(font), style.textBaseline);
        var textAlign = TEXT_ALIGN_TO_ANCHOR[style.textAlign]
            || style.textAlign;
        attr(textSvgEl, 'dominant-baseline', 'central');
        attr(textSvgEl, 'text-anchor', textAlign);
        attr(textSvgEl, 'x', x + '');
        attr(textSvgEl, 'y', y + '');
    }
};
export { svgText as text };
