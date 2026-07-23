import { isString } from '../core/util.js';
export function parseXML(svg) {
    if (isString(svg)) {
        var parser = new DOMParser();
        svg = parser.parseFromString(svg, 'text/xml');
    }
    var svgNode = svg;
    if (svgNode.nodeType === 9) {
        svgNode = svgNode.firstChild;
    }
    while (svgNode.nodeName.toLowerCase() !== 'svg' || svgNode.nodeType !== 1) {
        svgNode = svgNode.nextSibling;
    }
    return svgNode;
}
