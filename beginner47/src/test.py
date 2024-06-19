import click
from hashlib import sha256


@click.group()
def cli():
    pass


@cli.command()
@click.option("--name", prompt="input your name")
def hello(name):
    click.echo(f"hello {name}")


@cli.command()
@click.option("--password", prompt=True, hide_input=True)
def pw(password):
    m = sha256()
    m.update(password.encode())
    print(m.digest())


if __name__ == "__main__":
    cli()
