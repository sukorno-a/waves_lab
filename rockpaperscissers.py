import random
play_game = str(input("De you want play rock, paper, scissors?"))
if play_game == "Yes" or play_game == "yes":
    plays = int(input("Yay! How many times would you like to play?"))
else: 
    print("Aw. That's a shame. Maybe another time. Bye bye")
    plays = 0
plays_counter = 0
data = ["Rock", "Paper", "Scissors"]
while plays_counter < plays:
    guess = str(input("Type rock, paper or scissors!"))
    if guess == "Rock" or guess == "rock":
        guess = 1
    if guess == "Paper" or guess == "paper":
        guess = 2
    if guess == "Scissors" or guess == "scissors":
        guess = 3
    computer = random.randint(0,2)
    if (guess - 1) == computer:
        print(f"You drew! Oh well You both picked {data[computer]}!")
        plays_counter += 1
    elif (guess - 1) > computer or (guess == 1 and computer == 2):
        print(f"You won! Yay! The computer picked {data[computer]} :)")
        plays_counter += 1
    else:
        print(f"You lost! That's sad The computer picked {data[computer]} :(")
        plays_counter += 1


    