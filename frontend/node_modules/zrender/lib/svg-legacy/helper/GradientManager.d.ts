import Definable from './Definable';
import Displayable from '../../graphic/Displayable';
import { GradientObject } from '../../graphic/Gradient';
export default class GradientManager extends Definable {
    constructor(zrId: number, svgRoot: SVGElement);
    addWithoutUpdate(svgElement: SVGElement, displayable: Displayable): void;
    add(gradient: GradientObject): SVGElement;
    update(gradient: GradientObject | string): void;
    updateDom(gradient: GradientObject, dom: SVGElement): void;
    markUsed(displayable: Displayable): void;
}
