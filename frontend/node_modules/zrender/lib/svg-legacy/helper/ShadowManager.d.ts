import Definable from './Definable';
import Displayable from '../../graphic/Displayable';
export default class ShadowManager extends Definable {
    private _shadowDomMap;
    private _shadowDomPool;
    constructor(zrId: number, svgRoot: SVGElement);
    private _getFromPool;
    update(svgElement: SVGElement, displayable: Displayable): void;
    remove(svgElement: SVGElement, displayable: Displayable): void;
    updateDom(svgElement: SVGElement, displayable: Displayable, shadowDom: SVGElement): void;
    removeUnused(): void;
}
