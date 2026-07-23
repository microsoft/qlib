/**
 * @file Manages SVG gradient elements.
 * @author Zhang Wenli
 */

import Definable from './Definable';
import * as zrUtil from '../../core/util';
import Displayable from '../../graphic/Displayable';
import { GradientObject } from '../../graphic/Gradient';
import { getIdURL, isGradient, isLinearGradient, isRadialGradient, normalizeColor, round4 } from '../../svg/helper';
import { createElement } from '../../svg/core';


type GradientObjectExtended = GradientObject & {
    __dom: SVGElement
}

/**
 * Manages SVG gradient elements.
 *
 * @param   zrId    zrender instance id
 * @param   svgRoot root of SVG document
 */
export default class GradientManager extends Definable {

    constructor(zrId: number, svgRoot: SVGElement) {
        super(zrId, svgRoot, ['linearGradient', 'radialGradient'], '__gradient_in_use__');
    }


    /**
     * Create new gradient DOM for fill or stroke if not exist,
     * but will not update gradient if exists.
     *
     * @param svgElement   SVG element to paint
     * @param displayable  zrender displayable element
     */
    addWithoutUpdate(
        svgElement: SVGElement,
        displayable: Displayable
    ) {
        if (displayable && displayable.style) {
            const that = this;
            zrUtil.each(['fill', 'stroke'], function (fillOrStroke: 'fill' | 'stroke') {
                let value = displayable.style[fillOrStroke] as GradientObject;
                if (isGradient(value)) {
                    const gradient = value as GradientObjectExtended;
                    const defs = that.getDefs(true);

                    // Create dom in <defs> if not exists
                    let dom;
                    if (gradient.__dom) {
                        // Gradient exists
                        dom = gradient.__dom;
                        if (!defs.contains(gradient.__dom)) {
                            // __dom is no longer in defs, recreate
                            that.addDom(dom);
                        }
                    }
                    else {
                        // New dom
                        dom = that.add(gradient);
                    }

                    that.markUsed(displayable);

                    svgElement.setAttribute(fillOrStroke, getIdURL(dom.getAttribute('id')));
                }
            });
        }
    }


    /**
     * Add a new gradient tag in <defs>
     *
     * @param   gradient zr gradient instance
     */
    add(gradient: GradientObject): SVGElement {
        let dom;
        if (isLinearGradient(gradient)) {
            dom = createElement('linearGradient');
        }
        else if (isRadialGradient(gradient)) {
            dom = createElement('radialGradient');
        }
        else {
            if (process.env.NODE_ENV !== 'production') {
                zrUtil.logError('Illegal gradient type.');
            }
            return null;
        }

        // Set dom id with gradient id, since each gradient instance
        // will have no more than one dom element.
        // id may exists before for those dirty elements, in which case
        // id should remain the same, and other attributes should be
        // updated.
        gradient.id = gradient.id || this.nextId++;
        dom.setAttribute('id', 'zr' + this._zrId
            + '-gradient-' + gradient.id);

        this.updateDom(gradient, dom);
        this.addDom(dom);

        return dom;
    }


    /**
     * Update gradient.
     *
     * @param gradient zr gradient instance or color string
     */
    update(gradient: GradientObject | string) {
        if (!isGradient(gradient)) {
            return;
        }

        const that = this;
        this.doUpdate(gradient, function () {
            const dom = (gradient as GradientObjectExtended).__dom;
            if (!dom) {
                return;
            }

            const tagName = dom.tagName;
            const type = gradient.type;
            if (type === 'linear' && tagName === 'linearGradient'
                || type === 'radial' && tagName === 'radialGradient'
            ) {
                // Gradient type is not changed, update gradient
                that.updateDom(gradient, (gradient as GradientObjectExtended).__dom);
            }
            else {
                // Remove and re-create if type is changed
                that.removeDom(gradient);
                that.add(gradient);
            }
        });
    }


    /**
     * Update gradient dom
     *
     * @param gradient zr gradient instance
     * @param dom DOM to update
     */
    updateDom(gradient: GradientObject, dom: SVGElement) {
        if (isLinearGradient(gradient)) {
            dom.setAttribute('x1', gradient.x as any);
            dom.setAttribute('y1', gradient.y as any);
            dom.setAttribute('x2', gradient.x2 as any);
            dom.setAttribute('y2', gradient.y2 as any);
        }
        else if (isRadialGradient(gradient)) {
            dom.setAttribute('cx', gradient.x as any);
            dom.setAttribute('cy', gradient.y as any);
            dom.setAttribute('r', gradient.r as any);
        }
        else {
            if (process.env.NODE_ENV !== 'production') {
                zrUtil.logError('Illegal gradient type.');
            }
            return;
        }

        dom.setAttribute('gradientUnits',
            gradient.global
                ? 'userSpaceOnUse' // x1, x2, y1, y2 in range of 0 to canvas width or height
                : 'objectBoundingBox' // x1, x2, y1, y2 in range of 0 to 1
        );

        // Remove color stops if exists
        dom.innerHTML = '';

        // Add color stops
        const colors = gradient.colorStops;
        for (let i = 0, len = colors.length; i < len; ++i) {
            const stop = createElement('stop');
            stop.setAttribute('offset', round4(colors[i].offset) * 100 + '%');

            const stopColor = colors[i].color;
            // Fix Safari bug that stop-color not recognizing alpha #9014
            const {color, opacity} = normalizeColor(stopColor);
            // stop-color cannot be color, since:
            // The opacity value used for the gradient calculation is the
            // *product* of the value of stop-opacity and the opacity of the
            // value of stop-color.
            // See https://www.w3.org/TR/SVG2/pservers.html#StopOpacityProperty
            stop.setAttribute('stop-color', color);
            if (opacity < 1) {
                stop.setAttribute('stop-opacity', opacity as any);
            }

            dom.appendChild(stop);
        }

        // Store dom element in gradient, to avoid creating multiple
        // dom instances for the same gradient element
        (gradient as GradientObject as GradientObjectExtended).__dom = dom;
    }

    /**
     * Mark a single gradient to be used
     *
     * @param displayable displayable element
     */
    markUsed(displayable: Displayable) {
        if (displayable.style) {
            let gradient = displayable.style.fill as GradientObject as GradientObjectExtended;
            if (gradient && gradient.__dom) {
                super.markDomUsed(gradient.__dom);
            }

            gradient = displayable.style.stroke as GradientObject as GradientObjectExtended;
            if (gradient && gradient.__dom) {
                super.markDomUsed(gradient.__dom);
            }
        }
    }


}