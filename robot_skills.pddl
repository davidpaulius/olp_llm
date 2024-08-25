(define (domain object-level-plan)

    (:requirements :adl)

    (:types
        robot - object ; this will correspond to a robot in the scene (useful for mobile robot planning)
        container - object ; this refers to an object in which items can be "in" it
    )

    (:constants
        air - object ; this is a virtual object that signifies emptiness
        hand - object ; this refers to the robot's end-effector (for now, accounting for a single gripper)
        table - object ; this refers to the working surface of a robot (assuming to be table-top tasks)
        robot - robot ; this refers to the robot being used in problem-solving
    )

    (:predicates
        ; ************************************************************************************
        ; NOTE: object-centered predicates (task-level/geometric constraints):
        ; ************************************************************************************

        (at ?obj1 - object ?obj2 - object); this specifies that ?obj2 is at location near ?obj1
        (in ?obj1 - object ?obj2 - object) ; this specifies that ?obj2 is inside ?obj1
        (on ?obj1 - object ?obj2 - object) ; this specifies that ?obj2 is on top of ?obj1 (z-axis)
        (under ?obj1 - object ?obj2 - object) ; this specifies that ?obj2 is under ?obj1 (z-axis)

        (left-of ?obj1 - object ?obj2 - object); this specifies that ?obj2 is to the left of ?obj1 (x-axis/y-axis)
        (right-of ?obj1 - object ?obj2 - object) ; this specifies that ?obj2 is to the right of ?obj1 (x-axis/y-axis)
        (in-front ?obj1 - object ?obj2 - object); this specifies that ?obj2 is in front of ?obj1 (x-axis/y-axis)
        (behind ?obj1 - object ?obj2 - object) ; this specifies that ?obj2 is behind ?obj1 (x-axis/y-axis)

        ; ************************************************************************************
        ; NOTE: state of matter predicates (object-level/logical constraints):
        ; ************************************************************************************

        (is-whole ?obj - object) ; ?obj is a whole object (think of an object before it gets divided by cutting)
        (is-sliced ?obj - object) ; ?obj is in a sliced state (from whole to cut)
        (is-ground ?obj - object) ; ?obj is in a ground state (e.g. powder)
        (is-juiced ?obj - object) ; ?obj is in a juiced state
        (is-mixed ?obj - object) ; ?obj is in a mixed state (at task level, this refers to a container whose contents are mixed)

        ;(no-perception) ; this is a flag for planning without perception switched on (i.e., no simulation)
        (is-mobile) ; this is a flag for planning for mobile or non-mobile robots
    )

    (:action pick ; -- robot picks up ?obj that is on top of a surface ?surface by grasping from top
        :parameters (
            ?obj - object ; any object that can be grasped by the robot
            ?surface - object ; some kind of object or surface that lies under the object
        )
        :precondition (and
            ; NOTE: specifying cases for mobile and non-mobile robots:
            (or (at ?surface robot) (not (is-mobile)))

            (in hand air) ; there is "air" (nothing) in the robot's gripper/hand
            (on ?obj air) ; there is "air" (nothing) on the object (for collision-free picking)
            (under ?obj ?surface) (on ?surface ?obj) ; object is on top of a surface
        )
        :effect (and
            (in hand ?obj) ; there is now an object in the robot's hand
            (on ?obj hand) ; the robot's hand is now on top of the object (assuming pick from top)
            (under ?obj air) ; there is now "air" below the object (after being lifted from picking)
            (on ?surface air) ; there is "air" (nothing) on the surface

            (not (in hand air)) ; there is no longer "air" in the gripper
            (not (on ?obj air)) ; there is no longer "air" on top of the object
            (not (on ?surface ?obj)) (not (under ?obj ?surface)) ; the object is no longer on the surface
        )
    )

    (:action place-on-top ; -- robot releases object from hand: this will free the hand while placing the object on some surface
        :parameters (
            ?obj - object ; any object that can be grasped by the robot
            ?surface - object ; some kind of object or surface that is free for placement
        )
        :precondition (and
            ; NOTE: specifying cases for mobile and non-mobile robots:
            (or (at ?surface robot) (not (is-mobile)))

            (in hand ?obj) ; there is an object in the robot's hand
            (on ?obj hand) ; the robot's hand is on top of the object (assuming it was picked from top)
            (under ?obj air) ; there is "air" below the object (clear for placement)
            (on ?surface air) ; there is "air" (nothing) on the surface, making it clear for object placement
        )
        :effect (and
            (in hand air) ; there is now "air" (nothing) in the robot's gripper/hand
            (on ?obj air) ; there is  now "air" (nothing) on the object

            (on ?surface ?obj) (under ?obj ?surface) ; object is on top of a surface

            (not (in hand ?obj)) ; the gripper no longer contains the object
            (not (on ?obj hand)) ; the object no longer has the object on top of it
            (not (on ?surface air)) ; the surface is no longer free or empty
            (not (under ?obj air)) ; the object no longer has air below it, as the surface now sits below it
        )
    )

    (:action pour
        :parameters (?obj - object ?source - object ?target - object)
        :precondition (and
            (or
                (at ?target robot)
                (not (is-mobile))
            )

            ; source container (which is in the hand) initially contains some kind of object:
            (under ?source air)
            (in ?source ?obj)
            (on ?source ?obj)

            (under ?obj ?source)
            (in hand ?source)

            ; target container should be collision-free (not caring about what is inside):
            (on ?target air)
        )
        :effect (and
            (in ?source air)
            (in ?target ?obj)
            (under ?obj ?target)

            (not (in ?source ?obj))
            (not (on ?source ?obj))

            (not (in ?target air))
            (not (under ?obj ?source))
        )
    )

    (:action place-inside
        ; releasing object from hand - this will free the hand while placing the object on some surface:
        :parameters (
            ?obj - object ; any object that can be grasped by the robot
            ?container - container ; some kind of object that acts as a container
        )
        :precondition (and
            ; NOTE: specifying cases for mobile and non-mobile robots:
            (or (at ?container robot) (not (is-mobile)))

            (in hand ?obj) ; there is an object in the robot's hand
            (on ?obj hand) ; the robot's hand is on top of the object (assuming it was picked from top)
            (under ?obj air) ; there is "air" below the object (clear for placement)
            (on ?container air) ; there is "air" (nothing) on the container, making it clear for object placement
        )
        :effect (and
            (in hand air) ; there is now "air" (nothing) in the robot's gripper/hand
            (on ?obj air) ; there is  now "air" (nothing) on the object

            (in ?container ?obj) ; object is now inside the container

            (not (in hand ?obj)) ; the gripper no longer contains the object
            (not (on ?obj hand)) ; the object no longer has the object on top of it
            (not (in ?container air)) ; the container is no longer empty
            (not (under ?obj air)) ; the object no longer has air below it, as the container now sits below it
        )
    )

    (:action move-to ; -- robot moves from one location to another (locations are assumed to be surfaces or objects in scene)
        :parameters ( ?robot - robot ?loc1 - object ?loc2 - object
        )
        :precondition (and
            (at ?loc1 ?robot) (not (at ?loc2 ?robot)) ; the robot is at some other location and not at the destination
        )
        :effect (and
            (at ?loc2 ?robot) (not (at ?loc1 ?robot)) ; the robot is now at the destination and no longer at the previous place
        )
    )
)