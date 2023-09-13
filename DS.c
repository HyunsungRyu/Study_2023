/*
#include <stdio.h>
#include <stdlib.h>
#define MAX_STACK_SIZE 100
typedef int element;
element stack[MAX_STACK_SIZE];
int top = -1;

int is_empty() {
	if (top == -1) {
		return 1;
	}
	else {
		return 0;
	}
}
int is_full() {
	if (top == MAX_STACK_SIZE - 1) {
		return 1;
	}
	else {
		return 0;
	}
}
void push(element item) {
	if (is_full(item) == 1) {
		printf("error");
	}
	else {
		top += 1;
		stack[top] = item;
	}
}
element pop( ) {
	if (is_empty() == 1) {
		printf("error");
	}
	else {
		int a  = stack[top];
		top -= 1;
		return a;
	}
}
int main(void) {
	push(1);
	push(2);
	push(3);
	printf("%d\n", pop());
	printf("%d\n", pop());
	printf("%d\n", pop());
}
*/
#include <stdio.h>
#include <stdlib.h>

#define MAX_STACK_SIZE 100
typedef int element;
typedef struct {
	element data[MAX_STACK_SIZE];
	int top;
}StackType;

void init_stack(StackType* s)
{
	s->top = -1;
}

int is_empty(StackType* s)
{
	if (s->top == -1) {
		return 1;
	}
	else {
		return 0;
	}
}

int is_full(StackType* s)
{
	if (s->top == MAX_STACK_SIZE - 1) {
		return 1;
	}
	else {
		return 0;
	}
}

void push(StackType* s, element item)
{
	if (is_full(s)==1)
	{
		printf("error");
	}
	else {
		s->top +=1;
		s->data[s->top] = item;
	}
}

element pop(StackType* s)
{
	if (is_empty(s) == 1) {
		printf("error");
	}
	else {
		int a = s->data[s->top];
		s->top -= 1;
		return a;
	}
}
int main(void) {
	StackType s;

	init_stack(&s);
	push(&s, 1);
	push(&s, 2);
	push(&s, 3);
	printf("%d\n", pop(&s));
	printf("%d\n", pop(&s));
	printf("%d\n", pop(&s));
}