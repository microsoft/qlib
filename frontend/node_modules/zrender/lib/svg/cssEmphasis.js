import { liftColor } from '../tool/color.js';
import { getClassId } from './cssClassId.js';
export function createCSSEmphasis(el, attrs, scope) {
    if (!el.ignore) {
        if (el.isSilent()) {
            var style = {
                'pointer-events': 'none'
            };
            setClassAttribute(style, attrs, scope, true);
        }
        else {
            var emphasisStyle = el.states.emphasis && el.states.emphasis.style
                ? el.states.emphasis.style
                : {};
            var fill = emphasisStyle.fill;
            if (!fill) {
                var normalFill = el.style && el.style.fill;
                var selectFill = el.states.select
                    && el.states.select.style
                    && el.states.select.style.fill;
                var fromFill = el.currentStates.indexOf('select') >= 0
                    ? (selectFill || normalFill)
                    : normalFill;
                if (fromFill) {
                    fill = liftColor(fromFill);
                }
            }
            var lineWidth = emphasisStyle.lineWidth;
            if (lineWidth) {
                var scaleX = (!emphasisStyle.strokeNoScale && el.transform)
                    ? el.transform[0]
                    : 1;
                lineWidth = lineWidth / scaleX;
            }
            var style = {
                cursor: 'pointer'
            };
            if (fill) {
                style.fill = fill;
            }
            if (emphasisStyle.stroke) {
                style.stroke = emphasisStyle.stroke;
            }
            if (lineWidth) {
                style['stroke-width'] = lineWidth;
            }
            setClassAttribute(style, attrs, scope, true);
        }
    }
}
function setClassAttribute(style, attrs, scope, withHover) {
    var styleKey = JSON.stringify(style);
    var className = scope.cssStyleCache[styleKey];
    if (!className) {
        className = scope.zrId + '-cls-' + getClassId();
        scope.cssStyleCache[styleKey] = className;
        scope.cssNodes['.' + className + (withHover ? ':hover' : '')] = style;
    }
    attrs["class"] = attrs["class"] ? (attrs["class"] + ' ' + className) : className;
}
