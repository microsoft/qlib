/**
 * @file Manages SVG pattern elements.
 * @author Zhang Wenli
 */

import Definable from './Definable';
import * as zrUtil from '../../core/util';
import Displayable from '../../graphic/Displayable';
import {PatternObject} from '../../graphic/Pattern';
import {createOrUpdateImage} from '../../graphic/helper/image';
import WeakMap from '../../core/WeakMap';
import { getIdURL, isPattern, isSVGPattern } from '../../svg/helper';
import { createElement } from '../../svg/core';

const patternDomMap = new WeakMap<PatternObject, SVGElement>();

/**
 * Manages SVG pattern elements.
 *
 * @param   zrId    zrender instance id
 * @param   svgRoot root of SVG document
 */
export default class PatternManager extends Definable {

    constructor(zrId: number, svgRoot: SVGElement) {
        super(zrId, svgRoot, ['pattern'], '__pattern_in_use__');
    }


    /**
     * Create new pattern DOM for fill or stroke if not exist,
     * but will not update pattern if exists.
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
                const pattern = displayable.style[fillOrStroke] as PatternObject;
                if (isPattern(pattern)) {
                    const defs = that.getDefs(true);

                    // Create dom in <defs> if not exists
                    let dom = patternDomMap.get(pattern);
                    if (dom) {
                        // Pattern exists
                        if (!defs.contains(dom)) {
                            // __dom is no longer in defs, recreate
                            that.addDom(dom);
                        }
                    }
                    else {
                        // New dom
                        dom = that.add(pattern);
                    }

                    that.markUsed(displayable);

                    svgElement.setAttribute(fillOrStroke, getIdURL(dom.getAttribute('id')));
                }
            });
        }
    }


    /**
     * Add a new pattern tag in <defs>
     *
     * @param   pattern zr pattern instance
     */
    add(pattern: PatternObject): SVGElement {
        if (!isPattern(pattern)) {
            return;
        }

        let dom = createElement('pattern');

        pattern.id = pattern.id == null ? this.nextId++ : pattern.id;
        dom.setAttribute('id', 'zr' + this._zrId
            + '-pattern-' + pattern.id);

        dom.setAttribute('patternUnits', 'userSpaceOnUse');

        this.updateDom(pattern, dom);
        this.addDom(dom);

        return dom;
    }


    /**
     * Update pattern.
     *
     * @param pattern zr pattern instance or color string
     */
    update(pattern: PatternObject | string) {
        if (!isPattern(pattern)) {
            return;
        }

        const that = this;
        this.doUpdate(pattern, function () {
            const dom = patternDomMap.get(pattern);
            that.updateDom(pattern, dom);
        });
    }


    /**
     * Update pattern dom
     *
     * @param pattern zr pattern instance
     * @param patternDom DOM to update
     */
    updateDom(pattern: PatternObject, patternDom: SVGElement) {
        if (isSVGPattern(pattern)) {
            // New SVGPattern will not been supported in the legacy SVG renderer.
            // svg-legacy will been removed soon.

            // const svgElement = pattern.svgElement;
            // const isStringSVG = typeof svgElement === 'string';
            // if (isStringSVG || svgElement.parentNode !== patternDom) {
            //     if (isStringSVG) {
            //         patternDom.innerHTML = svgElement;
            //     }
            //     else {
            //         patternDom.innerHTML = '';
            //         patternDom.appendChild(svgElement);
            //     }

            //     patternDom.setAttribute('width', pattern.svgWidth as any);
            //     patternDom.setAttribute('height', pattern.svgHeight as any);
            // }
        }
        else {
            let img: SVGElement;
            const prevImage = patternDom.getElementsByTagName('image');
            if (prevImage.length) {
                if (pattern.image) {
                    // Update
                    img = prevImage[0];
                }
                else {
                    // Remove
                    patternDom.removeChild(prevImage[0]);
                    return;
                }
            }
            else if (pattern.image) {
                // Create
                img = createElement('image');
            }

            if (img) {
                let imageSrc;
                const patternImage = pattern.image;
                if (typeof patternImage === 'string') {
                    imageSrc = patternImage;
                }
                else if (patternImage instanceof HTMLImageElement) {
                    imageSrc = patternImage.src;
                }
                else if (patternImage instanceof HTMLCanvasElement) {
                    imageSrc = patternImage.toDataURL();
                }

                if (imageSrc) {
                    img.setAttribute('href', imageSrc);

                    // No need to re-render so dirty is empty
                    const hostEl = {
                        dirty: () => {}
                    };
                    const updateSize = (img: HTMLImageElement) => {
                        patternDom.setAttribute('width', img.width as any);
                        patternDom.setAttribute('height', img.height as any);
                    };

                    const createdImage = createOrUpdateImage(imageSrc, img as any, hostEl, updateSize);
                    if (createdImage && createdImage.width && createdImage.height) {
                        // Loaded before
                        updateSize(createdImage as HTMLImageElement);
                    }

                    patternDom.appendChild(img);
                }
            }
        }

        const x = pattern.x || 0;
        const y = pattern.y || 0;
        const rotation = (pattern.rotation || 0) / Math.PI * 180;
        const scaleX = pattern.scaleX || 1;
        const scaleY = pattern.scaleY || 1;
        const transform = `translate(${x}, ${y}) rotate(${rotation}) scale(${scaleX}, ${scaleY})`;
        patternDom.setAttribute('patternTransform', transform);
        patternDomMap.set(pattern, patternDom);
    }

    /**
     * Mark a single pattern to be used
     *
     * @param displayable displayable element
     */
    markUsed(displayable: Displayable) {
        if (displayable.style) {
            if (isPattern(displayable.style.fill)) {
                super.markDomUsed(patternDomMap.get(displayable.style.fill));
            }
            if (isPattern(displayable.style.stroke)) {
                super.markDomUsed(patternDomMap.get(displayable.style.stroke));
            }
        }
    }

}
