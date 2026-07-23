// TODO
// 1. shadow
// 2. Image: sx, sy, sw, sh

import {createElement, XLINKNS } from '../svg/core';
import { getMatrixStr, TEXT_ALIGN_TO_ANCHOR, adjustTextY } from '../svg/helper';
import * as matrix from '../core/matrix';
import Path, { PathStyleProps } from '../graphic/Path';
import ZRImage, { ImageStyleProps } from '../graphic/Image';
import { getLineHeight } from '../contain/text';
import TSpan, { TSpanStyleProps } from '../graphic/TSpan';
import SVGPathRebuilder from '../svg/SVGPathRebuilder';
import mapStyleToAttrs from '../svg/mapStyleToAttrs';
import { DEFAULT_FONT } from '../core/platform';

export interface SVGProxy<T> {
    brush(el: T): void
}

type AllStyleOption = PathStyleProps | TSpanStyleProps | ImageStyleProps;

function setTransform(svgEl: SVGElement, m: matrix.MatrixArray) {
    if (m) {
        attr(svgEl, 'transform', getMatrixStr(m));
    }
}

function attr(el: SVGElement, key: string, val: string | number) {
    if (!val || (val as any).type !== 'linear' && (val as any).type !== 'radial') {
        // Don't set attribute for gradient, since it need new dom nodes
        el.setAttribute(key, val as any);
    }
}

function attrXLink(el: SVGElement, key: string, val: string) {
    el.setAttributeNS(XLINKNS, key, val);
}

function attrXML(el: SVGElement, key: string, val: string) {
    el.setAttributeNS('http://www.w3.org/XML/1998/namespace', key, val);
}

function bindStyle(svgEl: SVGElement, style: PathStyleProps, el?: Path): void
function bindStyle(svgEl: SVGElement, style: TSpanStyleProps, el?: TSpan): void
function bindStyle(svgEl: SVGElement, style: ImageStyleProps, el?: ZRImage): void
function bindStyle(svgEl: SVGElement, style: AllStyleOption, el?: Path | TSpan | ZRImage) {
    mapStyleToAttrs((key, val) => attr(svgEl, key, val), style, el, true);
}

interface PathWithSVGBuildPath extends Path {
    __svgPathVersion: number
    __svgPathBuilder: SVGPathRebuilder
}

const svgPath: SVGProxy<Path> = {
    brush(el: Path) {
        const style = el.style;

        let svgEl = el.__svgEl;
        if (!svgEl) {
            svgEl = createElement('path');
            el.__svgEl = svgEl;
        }

        if (!el.path) {
            el.createPathProxy();
        }
        const path = el.path;

        if (el.shapeChanged()) {
            path.beginPath();
            el.buildPath(path, el.shape);
            el.pathUpdated();
        }

        const pathVersion = path.getVersion();
        const elExt = el as PathWithSVGBuildPath;
        let svgPathBuilder = elExt.__svgPathBuilder;
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

export {svgPath as path};

/***************************************************
 * IMAGE
 **************************************************/
const svgImage: SVGProxy<ZRImage> = {
    brush(el: ZRImage) {
        const style = el.style;
        let image = style.image;

        if (image instanceof HTMLImageElement) {
            image = image.src;
        }
        // heatmap layer in geo may be a canvas
        else if (image instanceof HTMLCanvasElement) {
            image = image.toDataURL();
        }
        if (!image) {
            return;
        }

        const x = style.x || 0;
        const y = style.y || 0;

        const dw = style.width;
        const dh = style.height;

        let svgEl = el.__svgEl;
        if (!svgEl) {
            svgEl = createElement('image');
            el.__svgEl = svgEl;
        }

        if (image !== el.__imageSrc) {
            attrXLink(svgEl, 'href', image as string);
            // Caching image src
            el.__imageSrc = image as string;
        }

        attr(svgEl, 'width', dw + '');
        attr(svgEl, 'height', dh + '');

        attr(svgEl, 'x', x + '');
        attr(svgEl, 'y', y + '');

        bindStyle(svgEl, style, el);
        setTransform(svgEl, el.transform);
    }
};
export {svgImage as image};

/***************************************************
 * TEXT
 **************************************************/


const svgText: SVGProxy<TSpan> = {
    brush(el: TSpan) {
        const style = el.style;

        let text = style.text;
        // Convert to string
        text != null && (text += '');
        if (!text || isNaN(style.x) || isNaN(style.y)) {
            return;
        }

        let textSvgEl = el.__svgEl as SVGTextElement;
        if (!textSvgEl) {
            textSvgEl = createElement('text') as SVGTextElement;
            attrXML(textSvgEl, 'xml:space', 'preserve');
            el.__svgEl = textSvgEl;
        }

        const font = style.font || DEFAULT_FONT;

        // style.font has been normalized by `normalizeTextStyle`.
        const textSvgElStyle = textSvgEl.style;
        textSvgElStyle.font = font;

        textSvgEl.textContent = text;

        bindStyle(textSvgEl, style, el);
        setTransform(textSvgEl, el.transform);

        // Consider different font display differently in vertial align, we always
        // set vertialAlign as 'middle', and use 'y' to locate text vertically.
        const x = style.x || 0;
        const y = adjustTextY(style.y || 0, getLineHeight(font), style.textBaseline);
        const textAlign = TEXT_ALIGN_TO_ANCHOR[style.textAlign as keyof typeof TEXT_ALIGN_TO_ANCHOR]
            || style.textAlign;

        attr(textSvgEl, 'dominant-baseline', 'central');
        attr(textSvgEl, 'text-anchor', textAlign);
        attr(textSvgEl, 'x', x + '');
        attr(textSvgEl, 'y', y + '');
    }
};
export {svgText as text};
