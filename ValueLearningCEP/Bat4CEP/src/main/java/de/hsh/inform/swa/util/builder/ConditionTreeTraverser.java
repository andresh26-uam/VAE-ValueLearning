package de.hsh.inform.swa.util.builder;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;
import java.util.concurrent.ThreadLocalRandom;
import de.hsh.inform.swa.cep.Attribute;
import de.hsh.inform.swa.cep.AttributeCondition;
import de.hsh.inform.swa.cep.AttributeOperator;
import de.hsh.inform.swa.cep.Condition;
import de.hsh.inform.swa.cep.Event;
import de.hsh.inform.swa.cep.EventCondition;
import de.hsh.inform.swa.cep.Rule;
import de.hsh.inform.swa.cep.TemplateEvent;
import de.hsh.inform.swa.cep.operators.attributes.arithmetic.ArithmeticOperator;
import de.hsh.inform.swa.cep.operators.attributes.logic.OrAttributeOperator;
import de.hsh.inform.swa.cep.operators.events.NotEventOperator;
import de.hsh.inform.swa.util.EventHandler;
/**
 * Class for traversing the ACT/ ECT.
 * @author Software Architecture Research Group
 *
 */
public class ConditionTreeTraverser {
    // adapted from: https://en.wikipedia.org/wiki/Tree_traversal#Pre-order_2
    public static Condition getConditionWithPreOrderIndex(Condition root, int id) {
        Stack<Condition> stack = new Stack<>();
        Condition currentCondition = root;
        int currentId = 0;
        while (currentId < id && (!stack.isEmpty() || currentCondition != null)) {
            if (currentCondition != null) {
                Condition[] children = currentCondition.getSubconditions();
                if (children == null) {
                    currentCondition = null;
                } else {
                    // push on stack in reverse order, so the leftmost is on top
                    for (int i = children.length - 1; i > 0; i--) {
                        stack.push(children[i]);
                    }
                    currentCondition = children[0];
                    currentId++;
                }
            } else {
                currentCondition = stack.pop();
                currentId++;
            }
        }
        return currentCondition;
    }

    public static <T extends Condition> T replaceNode(T originalTreeRoot,  T newSubTree, int insertionIndex) {
        if (insertionIndex == 0) {
            return newSubTree;
        }
        Node nodeToReplace = getNodeAtIndex(originalTreeRoot, insertionIndex);
        if (nodeToReplace.parent != null) {
            nodeToReplace.parent.setSubcondition(newSubTree, nodeToReplace.idxForParent);
        }
        return originalTreeRoot;
    }

    private static Node getNodeAtIndex(Condition treeRoot, int index) {
        Stack<Node> stack = new Stack<>();
        Node currentNode = new Node(treeRoot);
        int currentId = 0;
        while (currentId < index && (!stack.isEmpty() || currentNode != null)) {
            if (currentNode != null) {
                Condition[] operands = currentNode.current.getSubconditions();
                if (operands != null) {
                    for (int i = operands.length - 1; i >= 0; i--) {
                        Node operand = new Node(operands[i]);
                        operand.parent = currentNode.current;
                        operand.idxForParent = i;
                        stack.push(operand);
                    }
                }
            }
            currentNode = stack.pop();
            currentId++;
        }
        return currentNode;
    }

    public static Map<TemplateEvent, Integer> getTemplateEventsOccurrencesOfEct(Condition ect, EventHandler eh) {
        Map<TemplateEvent, Integer> templateEvents = new TreeMap<>();
        Stack<Condition> stack = new Stack<>();
        Condition currentCondition = ect;
        while (!stack.isEmpty() || currentCondition != null) {
            if (currentCondition != null) {
                if (currentCondition instanceof Event) {
                    TemplateEvent te = eh.getTemplateOfEvent((Event) currentCondition);
                    int eventTypeCount = templateEvents.getOrDefault(te, 0);
                    templateEvents.put(te, eventTypeCount + 1);
                }
                currentCondition = getNextCondition(currentCondition, stack);
            } else {
                currentCondition = stack.pop();
            }
        }
        return templateEvents;
    }

    public static Map<String, Integer> getEventTypeOccurrencesUnderEventCondition(Condition currentCondition) {
        Map<String, Integer> eventTypeOccurrences = new HashMap<>();
        Stack<Condition> stack = new Stack<>();
        while (!stack.isEmpty() || currentCondition != null) {
            if (currentCondition != null) {
                if (currentCondition instanceof Event) {
                    String type = ((Event) currentCondition).getType();
                    int occurrences = eventTypeOccurrences.getOrDefault(type, 0);
                    eventTypeOccurrences.put(type, occurrences + 1);
                }
                currentCondition = getNextCondition(currentCondition, stack);
            } else {
                currentCondition = stack.pop();
            }
        }
        return eventTypeOccurrences;
    }

    public static void simplfyACT(Rule rule, Condition act, int numberOfMaxDuplicates) {
        HashMap<String, Integer> duplicates = new HashMap<>();
        Stack<Node> stack = new Stack<>();

        while (!stack.isEmpty() || act != null) {
            if (act instanceof AttributeOperator) {

                if (!duplicates.containsKey(act.toString())) {
                    duplicates.put(act.toString(), 0);
                }

                if (duplicates.containsKey(act.toString())) {
                    duplicates.put(act.toString(), duplicates.get(act.toString()) + 1);
                }

                if (duplicates.get(act.toString()) > numberOfMaxDuplicates) {
                    Condition[] childs = stack.peek().parent.getSubconditions();

                    for (int i = 0; i < childs.length; i++) {
                        if (!childs[i].equals(act)) {
                            if (stack.peek().parentParent == null) {
                                rule.setAttributeConditionTreeRoot((AttributeCondition) childs[i]);
                            } else {
                                Condition parentParent = stack.peek().parentParent;
                                Condition[] parentParentChilds = parentParent.getSubconditions();
                                for (int y = 0; y < parentParentChilds.length; y++) {
                                    if (parentParentChilds[y].equals(stack.peek().parent)) {
                                        parentParent.setSubcondition(childs[i], y);
                                        break;
                                    }
                                }
                            }
                            break;
                        }
                    }
                }

                if (!stack.isEmpty()) {
                    stack.pop();
                }

            } else {
                Node prevNode = null;
                if (!stack.isEmpty())
                    prevNode = stack.pop();
                if (act instanceof AttributeCondition) {
                    Condition[] childs = act.getSubconditions();
                    for (int i = 0; i < childs.length; i++) {
                        Node node = new Node(childs[i]);
                        node.parent = act;
                        if (prevNode != null)
                            node.parentParent = prevNode.parent;
                        stack.push(node);
                    }
                }
            }

            if (!stack.isEmpty()) {
                act = stack.peek().current;
            } else {
                act = null;
            }
        }

    }
    public static void simplfyACTor(Rule rule, Condition act, int duplicates) {
        int countOrs = 0;

        Stack<Node> stack = new Stack<>();

        while (!stack.isEmpty() || act != null) {
            if (act instanceof OrAttributeOperator) {
                boolean alreadyPoped = false;
                countOrs++;

                if (countOrs > duplicates) {
                    Condition parent = stack.peek().parent;
                    Condition[] parentChilds = parent.getSubconditions();

                    for (int y = 0; y < parentChilds.length; y++) {
                        if (parentChilds[y].equals(act)) {
                            int index = ThreadLocalRandom.current().nextDouble() > 0.5 ? 1 : 0;

                            stack.pop();
                            alreadyPoped = true;

                            Condition con = act.getSubconditions()[index];
                            Node node = new Node(con);
                            node.parent = parent;
                            stack.push(node);

                            parent.setSubcondition(con, y);
                            break;
                        }
                    }

                }

                if (!alreadyPoped && !stack.isEmpty()) stack.pop();
                alreadyPoped = false;
            } else {
                Node prevNode = null;
                if (!stack.isEmpty()) prevNode = stack.pop();
                if (act instanceof AttributeCondition) {
                    Condition[] childs = act.getSubconditions();
                    if (childs != null) {
                        for (int i = 0; i < childs.length; i++) {
                            Node node = new Node(childs[i]);
                            node.parent = act;
                            if (prevNode != null) node.parentParent = prevNode.parent;
                            stack.push(node);
                        }
                    }
                }
            }

            if (!stack.isEmpty()) {
                act = stack.peek().current;
            } else {
                act = null;
            }
        }
    }


    public static void removeDuplicates(Rule rule, Condition ect, int numberOfMaxDuplicates) {
        HashMap<String, Integer> duplicates = new HashMap<>();
        Stack<Condition> stack = new Stack<>();
        Stack<Condition> breadcrumb = new Stack<>();
        while (!stack.isEmpty() || ect != null) {
            if (ect != null) {
                if (ect instanceof Event) {
                    String type = ((Event) ect).getType();
                    if (duplicates.containsKey(type) && duplicates.get(type) >= numberOfMaxDuplicates) {
                        Condition parent = breadcrumb.pop();
                        if (!breadcrumb.isEmpty()) { // parent == root
                            Condition parentParent = breadcrumb.peek();
                            Condition[] parentParentsChildren = parentParent.getSubconditions();

                            for (Condition child : parent.getSubconditions()) {
                                if (!child.equals(ect)) {
                                    for (int i = 0; i < parentParentsChildren.length; i++) {
                                        if (parentParentsChildren[i].equals(parent))
                                            parentParent.setSubcondition(child, i);
                                    }
                                }
                            }
                        } else {
                            for (Condition child : parent.getSubconditions()) {
                                if (!child.equals(ect)) {
                                    rule.setEventConditionTreeRoot((EventCondition) child);
                                }
                            }
                        }
                    } else {
                        if (!duplicates.containsKey(type)) {
                            duplicates.put(type, 0);
                        }
                        duplicates.put(type, duplicates.get(type) + 1);
                    }
                }
                if (!(ect instanceof Event))
                    breadcrumb.push(ect);
                ect = getNextCondition(ect, stack);
            } else {
                if (stack.size() == breadcrumb.size() - 1) {
                    breadcrumb.pop();
                }
                ect = stack.pop();
                if (stack.isEmpty() && ect != null) {
                    // insert a Placeholder for the right side of the tree
                    stack.push(null);
                }
            }
        }
    }

    public static Set<String> getAllEventAliasesUnderEventCondition(Condition ect) {
        Condition curCondition = ect;
        Set<String> eventsSoFar = new HashSet<>();
        Stack<Condition> stack = new Stack<>();
        
        while (!stack.isEmpty() || curCondition != null) {
            if (curCondition != null) {
                if (curCondition instanceof Event) {
                    String type = ((Event) curCondition).getType();
                    int number = 0;
                    String event = type + number;
                    while (eventsSoFar.contains(event)) {
                        number++;
                        event = type + number;
                    }
                    eventsSoFar.add(event);
                }
                curCondition = getNextCondition(curCondition, stack);
            } else {
                curCondition = stack.pop();
            }
        }

        return eventsSoFar;
    }

    private static Condition getNextCondition(Condition currentCondition, Stack<Condition> stack) {
        Condition[] children = currentCondition.getSubconditions();
        if(currentCondition instanceof NotEventOperator) children = new Condition[] {children[0]};
        if (children == null) return null;
        // push on stack in reverse order, so the leftmost is on top
        for (int i = children.length - 1; i > 0; i--) {
            stack.push(children[i]);
        }
        return children[0];
    }

    public static Set<String> getAllUsedAliases(Condition inCondition) {
        Condition currentCondition = inCondition;

        Set<String> usedAliases = new HashSet<>();
        Stack<Condition> stack = new Stack<>();

        while (!stack.isEmpty() || currentCondition != null) {
            if (currentCondition != null) {
                if (currentCondition instanceof AttributeOperator) {
                    AttributeOperator op = (AttributeOperator) currentCondition;
                    for(Attribute attr: op.getOperands()) {
                    	if(attr.getAlias() != null) {
                    		if(attr instanceof ArithmeticOperator) {
                    			Arrays.stream(((ArithmeticOperator)attr).getOperands()).forEach(sub -> usedAliases.add(sub.getAlias()));
                    		}else {
                    			usedAliases.add(attr.getAlias());
                    		}
                    	}
                    }
                }
                currentCondition = getNextCondition(currentCondition, stack);
            } else {
                currentCondition = stack.pop();
            }
        }

        return usedAliases;
    }

    public static Set<AttributeOperator> getAllAttributeComparisonOperatorsUsingBrokenAlias(Set<String> brokenAliases,
            Condition currentCondition) {
        Set<AttributeOperator> operatorsUsingBrokenAliases = new HashSet<>();
        Stack<Condition> stack = new Stack<>();
        int conflict=0;
        while (!stack.isEmpty() || currentCondition != null) {
            if (currentCondition != null) {
                if (currentCondition instanceof AttributeOperator) {
                    AttributeOperator op = (AttributeOperator) currentCondition;
                    Attribute[] operands = op.getOperands();
                    for (int i = 0; i < operands.length; i++) {
                    	boolean isArithmetic = false;
                    	if(operands[i] instanceof ArithmeticOperator) {
                    		isArithmetic = Arrays.stream(((ArithmeticOperator)operands[i]).getOperands()).map(x -> brokenAliases.contains(x.getAlias())).anyMatch(Boolean.TRUE::equals);
                    	}
                        if (isArithmetic || brokenAliases.contains(operands[i].getAlias())) {
                        	if(operatorsUsingBrokenAliases.contains(op)) {
                                op.setConflict(++conflict);
                            }
                            operatorsUsingBrokenAliases.add(op);
                            break;
                        }
                    }
                }
                currentCondition = getNextCondition(currentCondition, stack);
            } else {
                currentCondition = stack.pop();
            }
        }
        return operatorsUsingBrokenAliases;
    }
 
    private static class Node {
        Condition parent = null, current, parentParent = null;
        int idxForParent;

        Node(Condition current) {
            this.current = current;
        }
    }
}
