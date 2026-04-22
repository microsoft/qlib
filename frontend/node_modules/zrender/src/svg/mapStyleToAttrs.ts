
import Path, { DEFAULT_PATH_STYLE, PathStyleProps } from '../graphic/Path';
import ZRImage, { ImageStyleProps } from '../graphic/Image';
import TSpan, { TSpanStyleProps } from '../graphic/TSpan';
import { getLineDash } from '../canvas/dashStyle';
import { map } from '../core/util';
import { normalizeColor } from './helper';

type AllStyleOption = PathStyleProps | TSpanStyleProps | ImageStyleProps;

const NONE = 'none';
const mathRound = Math.round;

function pathHasFill(style: AllStyleOption): style is PathStyleProps {
    const fill = (style as PathStyleProps).fill;
    return fill != null && fill !== NONE;
}

function pathHasStroke(style: AllStyleOption): style is PathStyleProps {
    const stroke = (style as PathStyleProps).stroke;
    return stroke != null && stroke !== NONE;
}

const strokeProps = ['lineCap', 'miterLimit', 'lineJoin'] as const;
const svgStrokeProps = map(strokeProps, prop => `stroke-${prop.toLowerCase()}`);

export default function mapStyleToAttrs(
    updateAttr: (key: string, val: string | number) => void,
    style: AllStyleOption,
    el: Path | TSpan | ZRImage,
    /**
     * Will try not to set the attribute if it's using default value if not using forceUpdate.
     * Mainly for reduce the generated size in svg-ssr mode.
     */
    forceUpdate: boolean
): void {
    const opacity = style.opacity == null ? 1 : style.opacity;

    // only set opacity. stroke and fill cannot be applied to svg image
    if (el instanceof ZRImage) {
        updateAttr('opacity', opacity);
        return;
    }

    if (pathHasFill(style)) {
        const fill = normalizeColor(style.fill as string);
        updateAttr('fill', fill.color);
        const fillOpacity = style.fillOpacity != null
            ? style.fillOpacity * fill.opacity * opacity
            : fill.opacity * opacity;
        if (forceUpdate || fillOpacity < 1) {
            updateAttr('fill-opacity', fillOpacity);
        }
    }
    else {
        updateAttr('fill', NONE);
    }

    if (pathHasStroke(style)) {
        const stroke = normalizeColor(style.stroke as string);
        updateAttr('stroke', stroke.color);
        const strokeScale = style.strokeNoScale
            ? (el as Path).getLineScale()
            : 1;
        const strokeWidth = (strokeScale ? (style.lineWidth || 0) / strokeScale : 0);
        const strokeOpacity = style.strokeOpacity != null
            ? style.strokeOpacity * stroke.opacity * opacity
            : stroke.opacity * opacity;
        const strokeFirst = style.strokeFirst;

        if (forceUpdate || strokeWidth !== 1) {
            updateAttr('stroke-width', strokeWidth);
        }
        // stroke then fill for text; fill then stroke for others
        if (forceUpdate || strokeFirst) {
            updateAttr('paint-order', strokeFirst ? 'stroke' : 'fill');
        }
        if (forceUpdate || strokeOpacity < 1) {
            updateAttr('stroke-opacity', strokeOpacity);
        }

        if (style.lineDash) {
            let [lineDash, lineDashOffset] = getLineDash(el);
            if (lineDash) {
                lineDashOffset = mathRound(lineDashOffset || 0);
                updateAttr('stroke-dasharray', lineDash.join(','));
                if (lineDashOffset || forceUpdate) {
                    updateAttr('stroke-dashoffset', lineDashOffset);
                }
            }
        }
        else if (forceUpdate) {
            // Reset if force update.
            updateAttr('stroke-dasharray', NONE);
        }

        // PENDING reset
        for (let i = 0; i < strokeProps.length; i++) {
            const propName = strokeProps[i];
            if (forceUpdate || style[propName] !== DEFAULT_PATH_STYLE[propName]) {
                const val = style[propName] || DEFAULT_PATH_STYLE[propName];
                // TODO reset
                val && updateAttr(svgStrokeProps[i], val);
            }
        }
    }
    else if (forceUpdate) {
        updateAttr('stroke', NONE);
    }
}
