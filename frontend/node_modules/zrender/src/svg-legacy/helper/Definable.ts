/**
 * @file Manages elements that can be defined in <defs> in SVG,
 *       e.g., gradients, clip path, etc.
 * @author Zhang Wenli
 */

import {createElement} from '../../svg/core';
import * as zrUtil from '../../core/util';
import Displayable from '../../graphic/Displayable';


const MARK_UNUSED = '0';
const MARK_USED = '1';

/**
 * Manages elements that can be defined in <defs> in SVG,
 * e.g., gradients, clip path, etc.
 */
export default class Definable {

    nextId = 0

    protected _zrId: number
    protected _svgRoot: SVGElement
    protected _tagNames: string[]
    protected _markLabel: string
    protected _domName: string = '_dom'

    constructor(
        zrId: number,   // zrender instance id
        svgRoot: SVGElement,        // root of SVG document
        tagNames: string | string[],    // possible tag names
        markLabel: string,  // label name to make if the element
        domName?: string
    ) {
        this._zrId = zrId;
        this._svgRoot = svgRoot;
        this._tagNames = typeof tagNames === 'string' ? [tagNames] : tagNames;
        this._markLabel = markLabel;

        if (domName) {
            this._domName = domName;
        }
    }

    /**
     * Get the <defs> tag for svgRoot; optionally creates one if not exists.
     *
     * @param isForceCreating if need to create when not exists
     * @return SVG <defs> element, null if it doesn't
     * exist and isForceCreating is false
     */
    getDefs(isForceCreating?: boolean): SVGDefsElement {
        let svgRoot = this._svgRoot;
        let defs = this._svgRoot.getElementsByTagName('defs');
        if (defs.length === 0) {
            // Not exist
            if (isForceCreating) {
                let defs = svgRoot.insertBefore(
                    createElement('defs'), // Create new tag
                    svgRoot.firstChild // Insert in the front of svg
                ) as SVGDefsElement;
                if (!defs.contains) {
                    // IE doesn't support contains method
                    defs.contains = function (el) {
                        const children = defs.children;
                        if (!children) {
                            return false;
                        }
                        for (let i = children.length - 1; i >= 0; --i) {
                            if (children[i] === el) {
                                return true;
                            }
                        }
                        return false;
                    };
                }
                return defs;
            }
            else {
                return null;
            }
        }
        else {
            return defs[0];
        }
    }


    /**
     * Update DOM element if necessary.
     *
     * @param element style element. e.g., for gradient,
     *                                it may be '#ccc' or {type: 'linear', ...}
     * @param onUpdate update callback
     */
    doUpdate<T>(target: T, onUpdate?: (target: T) => void) {
        if (!target) {
            return;
        }

        const defs = this.getDefs(false);
        if ((target as any)[this._domName] && defs.contains((target as any)[this._domName])) {
            // Update DOM
            if (typeof onUpdate === 'function') {
                onUpdate(target);
            }
        }
        else {
            // No previous dom, create new
            const dom = this.add(target);
            if (dom) {
                (target as any)[this._domName] = dom;
            }
        }
    }

    add(target: any): SVGElement {
        return null;
    }

    /**
     * Add gradient dom to defs
     *
     * @param dom DOM to be added to <defs>
     */
    addDom(dom: SVGElement) {
        const defs = this.getDefs(true);
        if (dom.parentNode !== defs) {
            defs.appendChild(dom);
        }
    }


    /**
     * Remove DOM of a given element.
     *
     * @param target Target where to attach the dom
     */
    removeDom<T>(target: T) {
        const defs = this.getDefs(false);
        if (defs && (target as any)[this._domName]) {
            defs.removeChild((target as any)[this._domName]);
            (target as any)[this._domName] = null;
        }
    }


    /**
     * Get DOMs of this element.
     *
     * @return doms of this defineable elements in <defs>
     */
    getDoms() {
        const defs = this.getDefs(false);
        if (!defs) {
            // No dom when defs is not defined
            return [];
        }

        let doms: SVGElement[] = [];
        zrUtil.each(this._tagNames, function (tagName) {
            const tags = defs.getElementsByTagName(tagName) as HTMLCollectionOf<SVGElement>;
            // Note that tags is HTMLCollection, which is array-like
            // rather than real array.
            // So `doms.concat(tags)` add tags as one object.
            for (let i = 0; i < tags.length; i++) {
                doms.push(tags[i]);
            }
        });

        return doms;
    }


    /**
     * Mark DOMs to be unused before painting, and clear unused ones at the end
     * of the painting.
     */
    markAllUnused() {
        const doms = this.getDoms();
        const that = this;
        zrUtil.each(doms, function (dom) {
            (dom as any)[that._markLabel] = MARK_UNUSED;
        });
    }


    /**
     * Mark a single DOM to be used.
     *
     * @param dom DOM to mark
     */
    markDomUsed(dom: SVGElement) {
        dom && ((dom as any)[this._markLabel] = MARK_USED);
    };

    markDomUnused(dom: SVGElement) {
        dom && ((dom as any)[this._markLabel] = MARK_UNUSED);
    };

    isDomUnused(dom: SVGElement) {
        return dom && (dom as any)[this._markLabel] !== MARK_USED;
    }

    /**
     * Remove unused DOMs defined in <defs>
     */
    removeUnused() {
        const defs = this.getDefs(false);
        if (!defs) {
            // Nothing to remove
            return;
        }

        const doms = this.getDoms();
        zrUtil.each(doms, (dom) => {
            if (this.isDomUnused(dom)) {
                // Remove gradient
                defs.removeChild(dom);
            }
        });
    }

    /**
     * Get SVG element.
     *
     * @param displayable displayable element
     * @return SVG element
     */
    getSvgElement(displayable: Displayable): SVGElement {
        return displayable.__svgEl;
    }

}