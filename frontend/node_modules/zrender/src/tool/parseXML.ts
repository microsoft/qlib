import { isString } from '../core/util';

/**
 * For big svg string, this method might be time consuming.
 */
export function parseXML(svg: Document | string | SVGElement): SVGElement {
    if (isString(svg)) {
        const parser = new DOMParser();
        svg = parser.parseFromString(svg, 'text/xml');
    }
    let svgNode: Node = svg;
    // Document node. If using $.get, doc node may be input.
    if (svgNode.nodeType === 9) {
        svgNode = svgNode.firstChild;
    }
    // nodeName of <!DOCTYPE svg> is also 'svg'.
    while (svgNode.nodeName.toLowerCase() !== 'svg' || svgNode.nodeType !== 1) {
        svgNode = svgNode.nextSibling;
    }

    return svgNode as SVGElement;
}
