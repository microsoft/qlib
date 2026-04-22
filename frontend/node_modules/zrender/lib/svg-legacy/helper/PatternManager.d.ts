import Definable from './Definable';
import Displayable from '../../graphic/Displayable';
import { PatternObject } from '../../graphic/Pattern';
export default class PatternManager extends Definable {
    constructor(zrId: number, svgRoot: SVGElement);
    addWithoutUpdate(svgElement: SVGElement, displayable: Displayable): void;
    add(pattern: PatternObject): SVGElement;
    update(pattern: PatternObject | string): void;
    updateDom(pattern: PatternObject, patternDom: SVGElement): void;
    markUsed(displayable: Displayable): void;
}
