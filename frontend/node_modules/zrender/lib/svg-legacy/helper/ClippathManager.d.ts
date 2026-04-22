import Definable from './Definable';
import Displayable from '../../graphic/Displayable';
import Path from '../../graphic/Path';
export declare function hasClipPath(displayable: Displayable): boolean;
export default class ClippathManager extends Definable {
    private _refGroups;
    private _keyDuplicateCount;
    constructor(zrId: number, svgRoot: SVGElement);
    markAllUnused(): void;
    private _getClipPathGroup;
    update(displayable: Displayable, prevDisplayable: Displayable): SVGElement;
    updateDom(parentEl: SVGElement, clipPaths: Path[]): void;
    markUsed(displayable: Displayable): void;
    removeUnused(): void;
}
