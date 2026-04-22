import Displayable from '../../graphic/Displayable';
export default class Definable {
    nextId: number;
    protected _zrId: number;
    protected _svgRoot: SVGElement;
    protected _tagNames: string[];
    protected _markLabel: string;
    protected _domName: string;
    constructor(zrId: number, svgRoot: SVGElement, tagNames: string | string[], markLabel: string, domName?: string);
    getDefs(isForceCreating?: boolean): SVGDefsElement;
    doUpdate<T>(target: T, onUpdate?: (target: T) => void): void;
    add(target: any): SVGElement;
    addDom(dom: SVGElement): void;
    removeDom<T>(target: T): void;
    getDoms(): SVGElement[];
    markAllUnused(): void;
    markDomUsed(dom: SVGElement): void;
    markDomUnused(dom: SVGElement): void;
    isDomUnused(dom: SVGElement): boolean;
    removeUnused(): void;
    getSvgElement(displayable: Displayable): SVGElement;
}
